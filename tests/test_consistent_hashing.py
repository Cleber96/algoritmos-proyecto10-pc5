# tests/test_consistent_hashing.py
import pytest
import os
import sys
import hashlib # Para simular hashes de claves

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.consistent_hashing.consistent_hasher import ConsistentHasher
from src.common.utils import log_info, log_warning

@pytest.fixture
def empty_hasher():
    return ConsistentHasher(num_replicas=10) # Usar pocas réplicas para tests de distribución

@pytest.fixture
def populated_hasher():
    hasher = ConsistentHasher(num_replicas=10)
    hasher.add_node("node1")
    hasher.add_node("node2")
    hasher.add_node("node3")
    return hasher

class TestConsistentHasher:
    def test_initialization(self, empty_hasher):
        assert empty_hasher.num_replicas == 10
        assert not empty_hasher.nodes
        assert not empty_hasher.ring

    def test_add_node(self, empty_hasher):
        empty_hasher.add_node("node_A")
        assert "node_A" in empty_hasher.nodes
        assert len(empty_hasher.nodes) == 1
        assert len(empty_hasher.ring) == 10 # 1 node * 10 replicas

        empty_hasher.add_node("node_B")
        assert "node_B" in empty_hasher.nodes
        assert len(empty_hasher.nodes) == 2
        assert len(empty_hasher.ring) == 20 # 2 nodes * 10 replicas

    def test_add_duplicate_node_raises_error(self, populated_hasher):
        with pytest.raises(ValueError, match="Node 'node1' already exists in the hash ring."):
            populated_hasher.add_node("node1")

    def test_remove_node(self, populated_hasher):
        populated_hasher.remove_node("node2")
        assert "node2" not in populated_hasher.nodes
        assert len(populated_hasher.nodes) == 2 # node1, node3 remain
        assert len(populated_hasher.ring) == 20 # 2 nodes * 10 replicas

    def test_remove_non_existent_node_raises_error(self, empty_hasher):
        with pytest.raises(ValueError, match="Node 'non_existent_node' not found in the hash ring."):
            empty_hasher.remove_node("non_existent_node")
    
    def test_remove_only_node(self, empty_hasher):
        empty_hasher.add_node("sole_node")
        empty_hasher.remove_node("sole_node")
        assert not empty_hasher.nodes
        assert not empty_hasher.ring
        assert empty_hasher.ring_sorted_keys == []


    def test_get_node_with_no_nodes(self, empty_hasher):
        with pytest.raises(RuntimeError, match="No nodes available in the hash ring."):
            empty_hasher.get_node("some_key")

    def test_get_node_consistency(self, populated_hasher):
        key = "test_vector_key_123"
        assigned_node1 = populated_hasher.get_node(key)
        assigned_node2 = populated_hasher.get_node(key)
        assert assigned_node1 == assigned_node2

    def test_get_node_distribution(self, populated_hasher):
        # This is a probabilistic test, so we'll check for reasonable distribution
        # For num_replicas=10, 3 nodes, we expect some balance.
        num_keys = 1000
        key_assignments = {}
        for i in range(num_keys):
            key = f"vector_{i}"
            node = populated_hasher.get_node(key)
            key_assignments[node] = key_assignments.get(node, 0) + 1
        
        assert len(key_assignments) == 3 # Should assign to all 3 nodes
        
        # Check if the distribution is somewhat even. With 1000 keys and 3 nodes,
        # each node should get roughly 333 keys. Allow for 20% deviation.
        avg_keys_per_node = num_keys / len(populated_hasher.nodes)
        for node_id, count in key_assignments.items():
            assert count >= avg_keys_per_node * 0.8
            assert count <= avg_keys_per_node * 1.2
    
    def test_rebalancing_on_add_node(self, empty_hasher):
        empty_hasher.add_node("nodeA")
        empty_hasher.add_node("nodeB")
        
        keys = [f"key{i}" for i in range(20)]
        initial_assignments = {key: empty_hasher.get_node(key) for key in keys}

        # Add a new node
        empty_hasher.add_node("nodeC")
        
        # Check that some keys moved, but not all (consistent hashing property)
        moved_keys = 0
        for key in keys:
            if empty_hasher.get_node(key) != initial_assignments[key]:
                moved_keys += 1
        
        # With 20 keys and 10 replicas per node, adding a third node should
        # ideally reassign approx 1/N (1/3) of keys. With small N and replicas,
        # it might be less precise. We expect some movement but not all.
        assert 0 < moved_keys < len(keys) # Should not be 0, and not all

    def test_rebalancing_on_remove_node(self, populated_hasher):
        keys = [f"key{i}" for i in range(20)]
        initial_assignments = {key: populated_hasher.get_node(key) for key in keys}

        # Remove a node
        removed_node = "node2"
        populated_hasher.remove_node(removed_node)
        
        # Keys that were on 'node2' *must* move. Other keys should ideally stay.
        moved_keys = 0
        for key in keys:
            current_node = populated_hasher.get_node(key)
            if initial_assignments[key] == removed_node:
                assert current_node != removed_node # Must have moved
                moved_keys += 1
            elif current_node != initial_assignments[key]:
                moved_keys += 1
        
        assert moved_keys > 0 # At least the keys from the removed node must have moved
        # It's hard to assert an exact number for moved_keys because it's probabilistic,
        # but it should be significantly less than total_keys * (N-1)/N (bad for non-consistent hashing).
        # We only assert it's greater than 0, as some keys *must* move.