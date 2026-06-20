"""Consistent hash ring — implemented from scratch with no external libraries.

Uses SHA-256 hashing and virtual nodes for even key distribution.
The bisect module provides efficient sorted-list lookup.
"""

from __future__ import annotations

import bisect
import hashlib


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes for uniform distribution.

    Maps arbitrary string keys to a set of nodes (e.g. Qdrant shard URLs)
    using SHA-256 hashing and a configurable number of virtual nodes.
    """

    def __init__(self, nodes: list[str] | None = None, virtual_nodes: int = 150) -> None:
        """Initialize the hash ring.

        Args:
            nodes: Initial list of node identifiers.
            virtual_nodes: Number of virtual nodes per real node.
        """
        self._virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}
        self._sorted_keys: list[int] = []
        self._nodes: set[str] = set()

        for node in nodes or []:
            self.add_node(node)

    @staticmethod
    def _hash(key: str) -> int:
        """Compute a consistent hash for a key using SHA-256.

        Args:
            key: The string to hash.

        Returns:
            An integer hash value.
        """
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)

    @property
    def nodes(self) -> set[str]:
        """Return the set of real nodes in the ring."""
        return self._nodes.copy()

    @property
    def size(self) -> int:
        """Return the total number of points (virtual nodes) on the ring."""
        return len(self._sorted_keys)

    def add_node(self, node: str) -> None:
        """Add a node (with virtual nodes) to the ring.

        Args:
            node: The node identifier to add.
        """
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            hash_val = self._hash(virtual_key)
            self._ring[hash_val] = node
            bisect.insort(self._sorted_keys, hash_val)

    def remove_node(self, node: str) -> None:
        """Remove a node (and all its virtual nodes) from the ring.

        Args:
            node: The node identifier to remove.
        """
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self._virtual_nodes):
            virtual_key = f"{node}:vn{i}"
            hash_val = self._hash(virtual_key)
            if hash_val in self._ring:
                del self._ring[hash_val]
                idx = bisect.bisect_left(self._sorted_keys, hash_val)
                if idx < len(self._sorted_keys) and self._sorted_keys[idx] == hash_val:
                    self._sorted_keys.pop(idx)

    def get_node(self, key: str) -> str:
        """Find which node a key maps to on the ring.

        Uses binary search to find the first virtual node with a hash
        >= the key's hash, wrapping around if necessary.

        Args:
            key: The key to look up.

        Returns:
            The node identifier that owns this key.

        Raises:
            ValueError: If the ring is empty.
        """
        if not self._ring:
            raise ValueError("Hash ring is empty — no nodes available")

        hash_val = self._hash(key)
        idx = bisect.bisect_right(self._sorted_keys, hash_val)

        # Wrap around to the beginning of the ring
        if idx >= len(self._sorted_keys):
            idx = 0

        return self._ring[self._sorted_keys[idx]]

    def get_distribution(self) -> dict[str, int]:
        """Get the number of virtual nodes assigned to each real node.

        Returns:
            Dict mapping node name to virtual node count.
        """
        dist: dict[str, int] = {node: 0 for node in self._nodes}
        for node in self._ring.values():
            dist[node] = dist.get(node, 0) + 1
        return dist
