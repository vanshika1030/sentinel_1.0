"""Shard router — handles the write path for document upserts.

Routes documents to the correct Qdrant shard via the hash ring
and upserts the embedding vector with metadata.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client.models import PointStruct

from sentinel.sharding.shard_manager import ShardManager

logger = logging.getLogger(__name__)


class ShardRouter:
    """Routes document writes to the correct Qdrant shard."""

    def __init__(self, shard_manager: ShardManager) -> None:
        """Initialize the shard router.

        Args:
            shard_manager: The shard manager for client routing.
        """
        self._manager = shard_manager

    def _generate_doc_id(self, url: str, chunk_index: int) -> str:
        """Generate a deterministic document ID.

        Args:
            url: The source URL.
            chunk_index: The chunk index within the document.

        Returns:
            A UUID-style string ID.
        """
        seed = f"{url}::chunk::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    async def upsert(
        self,
        doc_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> str:
        """Upsert a single document into the correct shard.

        Args:
            doc_id: Unique document identifier.
            vector: Embedding vector.
            metadata: Document metadata (url, title, text, etc.).

        Returns:
            The shard URL the document was routed to.
        """
        shard_url = self._manager.get_shard_url(doc_id)
        client = self._manager.get_client(shard_url)

        point = PointStruct(
            id=doc_id,
            vector=vector,
            payload=metadata,
        )

        client.upsert(
            collection_name=self._manager.collection_name,
            points=[point],
        )
        logger.debug("Upserted doc %s to shard %s", doc_id, shard_url)
        return shard_url

    async def upsert_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Upsert a batch of documents, routing each to the correct shard.

        Args:
            documents: List of dicts with keys: doc_id, vector, metadata.

        Returns:
            Dict mapping shard URL to number of documents upserted.
        """
        # Group by shard
        shard_batches: dict[str, list[PointStruct]] = {}
        for doc in documents:
            doc_id = doc["doc_id"]
            shard_url = self._manager.get_shard_url(doc_id)
            if shard_url not in shard_batches:
                shard_batches[shard_url] = []
            shard_batches[shard_url].append(
                PointStruct(
                    id=doc_id,
                    vector=doc["vector"],
                    payload=doc["metadata"],
                )
            )

        # Upsert per shard
        counts: dict[str, int] = {}
        for shard_url, points in shard_batches.items():
            try:
                client = self._manager.get_client(shard_url)
                client.upsert(
                    collection_name=self._manager.collection_name,
                    points=points,
                )
                counts[shard_url] = len(points)
                logger.info("Batch upserted %d docs to %s", len(points), shard_url)
            except Exception as exc:
                logger.error("Batch upsert failed on %s: %s", shard_url, exc)
                counts[shard_url] = 0

        return counts
