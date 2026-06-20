"""Change detector — orchestrates the change detection pipeline.

Called after each successful crawl to detect, score, and store
content changes. Significant changes are embedded into the vector DB.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sentinel.core.chunker import ChunkerFactory
from sentinel.core.embedder import OpenAIEmbedder
from sentinel.ingestion.idempotency import IdempotencyGuard
from sentinel.ingestion.sources.base_source import FetchResult
from sentinel.sharding.shard_router import ShardRouter
from sentinel.temporal.change_scorer import ChangeScorer
from sentinel.temporal.differ import TextDiffer
from sentinel.temporal.snapshot_store import SnapshotStore
from sentinel.temporal.timeline import TimelineStore

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Orchestrates the change detection pipeline."""

    def __init__(
        self,
        snapshot_store: SnapshotStore,
        timeline_store: TimelineStore,
        change_scorer: ChangeScorer,
        embedder: OpenAIEmbedder,
        shard_router: ShardRouter,
        idempotency: IdempotencyGuard,
        chunker: ChunkerFactory | None = None,
    ) -> None:
        """Initialize the change detector.

        Args:
            snapshot_store: Store for content snapshots.
            timeline_store: Store for change records.
            change_scorer: Scorer for change significance.
            embedder: Embedding provider.
            shard_router: Router for vector DB upserts.
            idempotency: Guard for content deduplication.
            chunker: Text chunker factory.
        """
        self._snapshots = snapshot_store
        self._timeline = timeline_store
        self._scorer = change_scorer
        self._embedder = embedder
        self._router = shard_router
        self._idempotency = idempotency
        self._chunker = chunker or ChunkerFactory()
        self._differ = TextDiffer()

    async def process_fetch(
        self,
        fetch_result: FetchResult,
        competitor_name: str,
    ) -> dict[str, Any]:
        """Process a successful fetch: detect changes, score, store, embed.

        Args:
            fetch_result: The result from a content source fetch.
            competitor_name: Name of the competitor.

        Returns:
            Dict with processing results.
        """
        url = fetch_result.url
        content = fetch_result.content

        # Check idempotency
        if await self._idempotency.is_duplicate(url, content):
            logger.debug("Content unchanged (idempotent) for %s", url)
            return {"status": "skipped", "reason": "duplicate"}

        # Get previous snapshot
        prev_snapshot = await self._snapshots.get_previous_snapshot(url)

        # Save new snapshot
        snapshot = await self._snapshots.save_snapshot(
            url=url,
            content_text=content,
            competitor_name=competitor_name,
        )

        # Mark as processed for idempotency
        await self._idempotency.mark_processed(url, content)

        # If no previous snapshot, this is the first crawl — just embed
        if prev_snapshot is None:
            await self._embed_content(url, content, competitor_name, fetch_result)
            return {"status": "initial", "snapshot_id": snapshot.id}

        # Compute diff
        diff_result = self._differ.compute_diff(
            prev_snapshot.content_text, content
        )

        if diff_result.diff_ratio < 0.01:
            logger.debug("Negligible change for %s (ratio=%.4f)", url, diff_result.diff_ratio)
            return {"status": "unchanged", "diff_ratio": diff_result.diff_ratio}

        # Compute semantic similarity (optional, may fail)
        semantic_sim = None
        try:
            old_emb = await self._embedder.embed_text(prev_snapshot.content_text[:2000])
            new_emb = await self._embedder.embed_text(content[:2000])
            # Cosine similarity
            dot = sum(a * b for a, b in zip(old_emb, new_emb))
            norm_old = sum(a * a for a in old_emb) ** 0.5
            norm_new = sum(b * b for b in new_emb) ** 0.5
            if norm_old > 0 and norm_new > 0:
                semantic_sim = dot / (norm_old * norm_new)
        except Exception as exc:
            logger.warning("Failed to compute semantic similarity: %s", exc)

        # Score significance
        score_result = self._scorer.score(
            diff_ratio=diff_result.diff_ratio,
            diff_text=diff_result.diff_text,
            semantic_similarity=semantic_sim,
        )

        # Store change record
        change = await self._timeline.save_change(
            url=url,
            competitor_name=competitor_name,
            diff_text=diff_result.diff_text,
            significance_score=score_result.score,
            is_significant=score_result.is_significant,
        )

        # If significant, embed the diff text
        if score_result.is_significant:
            await self._embed_content(
                url, diff_result.diff_text, competitor_name, fetch_result,
                is_diff=True,
            )

        # Always embed the new content
        await self._embed_content(url, content, competitor_name, fetch_result)

        return {
            "status": "changed",
            "change_id": change.id,
            "significance": score_result.score,
            "is_significant": score_result.is_significant,
            "keywords": score_result.triggered_keywords,
        }

    async def _embed_content(
        self,
        url: str,
        content: str,
        competitor_name: str,
        fetch_result: FetchResult,
        is_diff: bool = False,
    ) -> None:
        """Chunk and embed content into the vector DB.

        Args:
            url: Source URL.
            content: Text content to embed.
            competitor_name: Competitor name.
            fetch_result: Original fetch result for metadata.
            is_diff: Whether this is diff text (vs full content).
        """
        chunks = self._chunker.split_text(content)
        if not chunks:
            return

        embeddings = await self._embedder.embed_batch(chunks)

        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{url}::{'diff' if is_diff else 'content'}::chunk::{i}",
            ))
            documents.append({
                "doc_id": doc_id,
                "vector": embedding,
                "metadata": {
                    "url": url,
                    "title": fetch_result.title,
                    "competitor_name": competitor_name,
                    "source_type": fetch_result.source_type,
                    "text": chunk,
                    "chunk_index": i,
                    "is_diff": is_diff,
                },
            })

        await self._router.upsert_batch(documents)
        logger.info(
            "Embedded %d chunks for %s (%s)",
            len(documents),
            url,
            "diff" if is_diff else "content",
        )
