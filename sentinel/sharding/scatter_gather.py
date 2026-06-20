"""Scatter-gather query — fans out searches across all healthy shards.

Sends vector search requests to all healthy Qdrant shards concurrently,
merges results by score, deduplicates by document ID, and returns
global top-K results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sentinel.core.config import settings
from sentinel.core.retriever import RetrievedDocument
from sentinel.sharding.shard_manager import ShardManager

logger = logging.getLogger(__name__)


class ScatterGather:
    """Scatter search across shards, gather and merge results."""

    def __init__(
        self,
        shard_manager: ShardManager,
        timeout: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """Initialize scatter-gather.

        Args:
            shard_manager: The shard manager with healthy shard info.
            timeout: Timeout per shard query in seconds.
            top_k: Number of global top results to return.
        """
        self._manager = shard_manager
        self._timeout = timeout or settings.qdrant_search_timeout
        self._top_k = top_k or settings.qdrant_top_k

    async def _search_shard(
        self,
        shard_url: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Search a single shard.

        Args:
            shard_url: URL of the Qdrant shard.
            query_vector: The query embedding vector.
            top_k: Number of results per shard.

        Returns:
            List of retrieved documents from this shard.
        """
        try:
            client = self._manager.get_client(shard_url)
            results = client.search(
                collection_name=self._manager.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            documents = []
            for hit in results:
                payload = hit.payload or {}
                documents.append(
                    RetrievedDocument(
                        doc_id=str(hit.id),
                        text=payload.get("text", ""),
                        metadata=payload,
                        score=hit.score,
                    )
                )
            return documents
        except Exception as exc:
            logger.error("Search failed on shard %s: %s", shard_url, exc)
            return []

    async def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        filter_params: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Scatter search across all healthy shards and gather results.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of global top results.
            filter_params: Optional Qdrant filter parameters.

        Returns:
            Merged, deduplicated, score-sorted list of documents.
        """
        top_k = top_k or self._top_k
        healthy_urls = self._manager.get_all_healthy_shard_urls()

        if not healthy_urls:
            logger.error("No healthy shards available for search")
            return []

        # Fan out to all healthy shards concurrently
        tasks = [
            asyncio.wait_for(
                self._search_shard(url, query_vector, top_k),
                timeout=self._timeout,
            )
            for url in healthy_urls
        ]

        results_per_shard = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate
        seen_ids: set[str] = set()
        all_docs: list[RetrievedDocument] = []

        for i, result in enumerate(results_per_shard):
            if isinstance(result, Exception):
                logger.warning(
                    "Shard %s timed out or errored: %s",
                    healthy_urls[i],
                    result,
                )
                continue
            for doc in result:
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    all_docs.append(doc)

        # Sort by score descending and return top-K
        all_docs.sort(key=lambda d: d.score, reverse=True)
        logger.info(
            "Scatter-gather: %d shards queried, %d total docs, returning top %d",
            len(healthy_urls),
            len(all_docs),
            min(top_k, len(all_docs)),
        )
        return all_docs[:top_k]
