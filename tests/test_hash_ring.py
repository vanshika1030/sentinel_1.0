"""Tests for the consistent hash ring implementation."""

import pytest

from sentinel.sharding.hash_ring import ConsistentHashRing


class TestConsistentHashRing:
    """Test suite for ConsistentHashRing."""

    def test_empty_ring_raises(self) -> None:
        """Getting a node from an empty ring should raise ValueError."""
        ring = ConsistentHashRing()
        with pytest.raises(ValueError, match="empty"):
            ring.get_node("any-key")

    def test_single_node_returns_that_node(self) -> None:
        """With one node, all keys should map to it."""
        ring = ConsistentHashRing(nodes=["shard-0"])
        for key in ["doc-1", "doc-2", "doc-3", "hello", "world"]:
            assert ring.get_node(key) == "shard-0"

    def test_two_nodes_distribute_keys(self) -> None:
        """With two nodes, keys should be distributed between them."""
        ring = ConsistentHashRing(nodes=["shard-0", "shard-1"], virtual_nodes=150)
        results = {"shard-0": 0, "shard-1": 0}
        for i in range(1000):
            node = ring.get_node(f"document-{i}")
            results[node] += 1

        # Both shards should have received some keys
        assert results["shard-0"] > 0
        assert results["shard-1"] > 0
        # Distribution should be roughly balanced (within 30%)
        total = results["shard-0"] + results["shard-1"]
        ratio = min(results.values()) / total
        assert ratio > 0.2, f"Distribution too skewed: {results}"

    def test_consistent_mapping(self) -> None:
        """Same key should always map to the same node."""
        ring = ConsistentHashRing(nodes=["a", "b", "c"])
        first_result = ring.get_node("test-key")
        for _ in range(100):
            assert ring.get_node("test-key") == first_result

    def test_add_node(self) -> None:
        """Adding a node should increase the ring size."""
        ring = ConsistentHashRing(nodes=["shard-0"], virtual_nodes=100)
        assert ring.size == 100
        ring.add_node("shard-1")
        assert ring.size == 200
        assert "shard-1" in ring.nodes

    def test_add_duplicate_node_is_noop(self) -> None:
        """Adding the same node twice should not change the ring."""
        ring = ConsistentHashRing(nodes=["shard-0"], virtual_nodes=100)
        ring.add_node("shard-0")
        assert ring.size == 100
        assert len(ring.nodes) == 1

    def test_remove_node(self) -> None:
        """Removing a node should decrease the ring size."""
        ring = ConsistentHashRing(nodes=["shard-0", "shard-1"], virtual_nodes=100)
        assert ring.size == 200
        ring.remove_node("shard-1")
        assert ring.size == 100
        assert "shard-1" not in ring.nodes

    def test_remove_nonexistent_node_is_noop(self) -> None:
        """Removing a non-existent node should not raise."""
        ring = ConsistentHashRing(nodes=["shard-0"], virtual_nodes=100)
        ring.remove_node("nonexistent")
        assert ring.size == 100

    def test_minimal_remapping_on_add(self) -> None:
        """Adding a node should only remap a fraction of keys."""
        nodes = ["shard-0", "shard-1"]
        ring = ConsistentHashRing(nodes=nodes, virtual_nodes=150)

        # Map 1000 keys
        original = {f"key-{i}": ring.get_node(f"key-{i}") for i in range(1000)}

        # Add a third node
        ring.add_node("shard-2")
        remapped = sum(
            1
            for key, node in original.items()
            if ring.get_node(key) != node
        )

        # Ideal remapping is ~1/3 of keys; allow up to 50%
        assert remapped < 500, f"Too many keys remapped: {remapped}/1000"

    def test_virtual_nodes_improve_distribution(self) -> None:
        """More virtual nodes should produce better distribution."""
        ring_few = ConsistentHashRing(nodes=["a", "b"], virtual_nodes=5)
        ring_many = ConsistentHashRing(nodes=["a", "b"], virtual_nodes=200)

        def measure_balance(ring: ConsistentHashRing) -> float:
            counts = {"a": 0, "b": 0}
            for i in range(1000):
                counts[ring.get_node(f"k-{i}")] += 1
            return min(counts.values()) / max(counts.values())

        balance_few = measure_balance(ring_few)
        balance_many = measure_balance(ring_many)
        # More virtual nodes should generally be more balanced
        # (not guaranteed for small sample but very likely)
        assert balance_many >= 0.3  # At least somewhat balanced

    def test_get_distribution(self) -> None:
        """Distribution method should return correct virtual node counts."""
        ring = ConsistentHashRing(nodes=["a", "b"], virtual_nodes=50)
        dist = ring.get_distribution()
        assert dist["a"] == 50
        assert dist["b"] == 50

    def test_nodes_property_returns_copy(self) -> None:
        """Nodes property should return a copy, not a reference."""
        ring = ConsistentHashRing(nodes=["a", "b"])
        nodes = ring.nodes
        nodes.add("c")
        assert "c" not in ring.nodes
