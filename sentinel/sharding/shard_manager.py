"""Shard manager — maintains Qdrant client connections and health state.

Holds the consistent hash ring and provides client routing plus
health monitoring for each Qdrant shard.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sentinel.core.config import settings
from sentinel.sharding.hash_ring import ConsistentHashRing

logger = logging.getLogger(__name__)


class ShardManager:
    """Manages Qdrant shard connections, health, and routing."""

    def __init__(
        self,
        qdrant_urls: list[str] | None = None,
        collection_name: str | None = None,
        virtual_nodes: int | None = None,
    ) -> None:
        """Initialize the shard manager.

        Args:
            qdrant_urls: List of Qdrant instance URLs.
            collection_name: Name of the Qdrant collection.
            virtual_nodes: Virtual nodes per shard in the hash ring.
        """
        self._urls = qdrant_urls or settings.qdrant_urls
        self._collection_name = collection_name or settings.qdrant_collection
        self._virtual_nodes = virtual_nodes or settings.qdrant_virtual_nodes

        self._hash_ring = ConsistentHashRing(
            nodes=self._urls, virtual_nodes=self._virtual_nodes
        )
        self._clients: dict[str, QdrantClient] = {}
        self._healthy: dict[str, bool] = {}

        for url in self._urls:
            self._clients[url] = QdrantClient(url=url, timeout=settings.qdrant_search_timeout)
            self._healthy[url] = True

    @property
    def hash_ring(self) -> ConsistentHashRing:
        """Return the underlying hash ring."""
        return self._hash_ring

    @property
    def collection_name(self) -> str:
        """Return the Qdrant collection name."""
        return self._collection_name

    def get_client(self, url: str) -> QdrantClient:
        """Get the Qdrant client for a specific URL.

        Args:
            url: The Qdrant instance URL.

        Returns:
            The QdrantClient for that shard.

        Raises:
            KeyError: If the URL is not a known shard.
        """
        if url not in self._clients:
            raise KeyError(f"Unknown shard URL: {url}")
        return self._clients[url]

    def get_shard_client(self, doc_id: str) -> QdrantClient:
        """Route a document ID to the correct shard client via the hash ring.

        Args:
            doc_id: The document identifier.

        Returns:
            The QdrantClient for the shard that owns this document.
        """
        url = self._hash_ring.get_node(doc_id)
        return self._clients[url]

    def get_shard_url(self, doc_id: str) -> str:
        """Get the shard URL for a given document ID.

        Args:
            doc_id: The document identifier.

        Returns:
            The URL of the shard that owns this document.
        """
        return self._hash_ring.get_node(doc_id)

    async def health_check(self) -> dict[str, bool]:
        """Ping each shard and update health status.

        Returns:
            Dict mapping shard URL to health status.
        """
        for url, client in self._clients.items():
            try:
                # Qdrant's health check
                client.get_collections()
                self._healthy[url] = True
            except Exception as exc:
                logger.warning("Shard %s unhealthy: %s", url, exc)
                self._healthy[url] = False
        return self._healthy.copy()

    def get_all_healthy_shards(self) -> list[QdrantClient]:
        """Return clients for all healthy shards.

        Returns:
            List of QdrantClient instances for healthy shards.
        """
        return [
            self._clients[url]
            for url, healthy in self._healthy.items()
            if healthy
        ]

    def get_all_healthy_shard_urls(self) -> list[str]:
        """Return URLs for all healthy shards.

        Returns:
            List of URLs for healthy shards.
        """
        return [url for url, healthy in self._healthy.items() if healthy]

    async def ensure_collection(self) -> None:
        """Create the collection on all shards if it doesn't exist."""
        for url, client in self._clients.items():
            try:
                collections = client.get_collections().collections
                exists = any(c.name == self._collection_name for c in collections)
                if not exists:
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dimensions,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info("Created collection '%s' on %s", self._collection_name, url)
            except Exception as exc:
                logger.error("Failed to ensure collection on %s: %s", url, exc)
