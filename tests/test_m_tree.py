# tests/test_m_tree.py
import pytest
import numpy as np
import random
import os
import sys

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.m_tree.m_tree import MTree
from src.common.models import Vector, SearchResult
from src.common.utils import get_distance_metric, euclidean_distance, cosine_similarity # Importar las funciones base también

# Fixture para un M-Tree vacío con métrica euclidiana
@pytest.fixture
def empty_m_tree_euclidean():
    return MTree(max_children=4, min_children=2, distance_metric="euclidean")

# Fixture para un M-Tree vacío con métrica coseno
@pytest.fixture
def empty_m_tree_cosine():
    return MTree(max_children=4, min_children=2, distance_metric="cosine")

# Fixture para un M-Tree poblado con datos simples
@pytest.fixture
def populated_m_tree_euclidean():
    tree = MTree(max_children=4, min_children=2, distance_metric="euclidean")
    vectors = [
        Vector("v1", [1.0, 1.0]),
        Vector("v2", [2.0, 2.0]),
        Vector("v3", [1.1, 1.0]),
        Vector("v4", [5.0, 5.0]),
        Vector("v5", [5.1, 5.2]),
        Vector("v6", [10.0, 10.0]),
        Vector("v7", [10.1, 9.9]),
        Vector("v8", [0.5, 0.5]),
        Vector("v9", [7.0, 7.0]),
        Vector("v10", [7.1, 6.9]),
        Vector("v11", [2.5, 2.5]),
        Vector("v12", [6.0, 6.0]),
        Vector("v13", [8.0, 8.0]),
        Vector("v14", [3.0, 3.0]),
        Vector("v15", [4.0, 4.0]),
    ]
    for vec in vectors:
        tree.insert(vec)
    return tree

@pytest.fixture
def populated_m_tree_cosine():
    tree = MTree(max_children=4, min_children=2, distance_metric="cosine")
    vectors = [
        Vector("c1", [1.0, 0.0]), # Angle 0
        Vector("c2", [0.0, 1.0]), # Angle 90
        Vector("c3", [1.0, 1.0]), # Angle 45
        Vector("c4", [-1.0, 0.0]), # Angle 180
        Vector("c5", [0.5, 0.5]), # Angle 45, magnitude different
        Vector("c6", [0.0, -1.0]) # Angle 270
    ]
    for vec in vectors:
        tree.insert(vec)
    return tree


class TestMTree:
    def test_initialization(self, empty_m_tree_euclidean, empty_m_tree_cosine):
        assert empty_m_tree_euclidean.root is None
        assert empty_m_tree_euclidean.size() == 0
        assert empty_m_tree_euclidean.distance_metric_name == "euclidean"
        assert empty_m_tree_euclidean.distance_fn == euclidean_distance # Check if it's the raw function or lambda

        assert empty_m_tree_cosine.root is None
        assert empty_m_tree_cosine.size() == 0
        assert empty_m_tree_cosine.distance_metric_name == "cosine"
        # For cosine, it should be a lambda, so we can't directly compare to cosine_similarity
        # We'll rely on functional tests for correctness.

    def test_insert_single_vector(self, empty_m_tree_euclidean):
        vec = Vector("v1", [1.0, 2.0])
        empty_m_tree_euclidean.insert(vec)
        assert empty_m_tree_euclidean.size() == 1
        assert empty_m_tree_euclidean.root is not None
        assert empty_m_tree_euclidean.root.is_leaf
        assert len(empty_m_tree_euclidean.root.entries) == 1
        assert empty_m_tree_euclidean.root.entries[0].vector.id == "v1"

    def test_insert_multiple_vectors_no_split(self, empty_m_tree_euclidean):
        vectors = [Vector(f"v{i}", [float(i), float(i)]) for i in range(1, 4)] # max_children is 4
        for vec in vectors:
            empty_m_tree_euclidean.insert(vec)
        assert empty_m_tree_euclidean.size() == 3
        assert empty_m_tree_euclidean.root.is_leaf
        assert len(empty_m_tree_euclidean.root.entries) == 3

    def test_insert_triggers_leaf_split(self, empty_m_tree_euclidean):
        # Insert enough vectors to force a leaf split (max_children=4, so 5th insert should split)
        vectors = [Vector(f"v{i}", [float(i), float(i)]) for i in range(1, 6)]
        for vec in vectors:
            empty_m_tree_euclidean.insert(vec)
        
        assert empty_m_tree_euclidean.size() == 5
        assert not empty_m_tree_euclidean.root.is_leaf # Root should now be an internal node
        assert len(empty_m_tree_euclidean.root.entries) == 2 # Should have 2 children (new leaves)

    def test_insert_triggers_internal_split_and_root_split(self, empty_m_tree_euclidean):
        # Populating with many vectors to trigger multiple splits and increase height
        vectors = [Vector(f"v{i}", [random.random() * 100, random.random() * 100]) for i in range(1, 50)]
        for vec in vectors:
            empty_m_tree_euclidean.insert(vec)
        
        assert empty_m_tree_euclidean.size() == 49
        assert not empty_m_tree_euclidean.root.is_leaf
        assert empty_m_tree_euclidean.root.get_height() > 1 # Should be at least 2 or more

    def test_search_knn_empty_tree(self, empty_m_tree_euclidean):
        query_vec = Vector("q", [0.0, 0.0])
        results = empty_m_tree_euclidean.search_knn(query_vec, k=1)
        assert len(results) == 0

    def test_search_knn_k_larger_than_tree_size(self, populated_m_tree_euclidean):
        query_vec = Vector("q", [0.0, 0.0])
        results = populated_m_tree_euclidean.search_knn(query_vec, k=100)
        assert len(results) == populated_m_tree_euclidean.size()
        # Verify results are sorted by distance
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i+1].distance

    def test_search_knn_euclidean_correctness(self, populated_m_tree_euclidean):
        # A simple query near existing vectors
        query_vec = Vector("q_near_v1", [1.05, 1.05])
        results = populated_m_tree_euclidean.search_knn(query_vec, k=3)
        
        assert len(results) == 3
        
        # Expected results sorted by distance (assuming Euclidean)
        # v1: [1.0, 1.0], dist = sqrt((0.05)^2 + (0.05)^2) = sqrt(0.005) = 0.0707
        # v3: [1.1, 1.0], dist = sqrt((-0.05)^2 + (0.05)^2) = sqrt(0.005) = 0.0707
        # v2: [2.0, 2.0], dist = sqrt((0.95)^2 + (0.95)^2) = sqrt(1.805) = 1.3435
        
        # Note: If v1 and v3 have same distance, order might vary.
        # We just check if they are among the top K and distances are correct.
        
        result_ids = {r.vector.id for r in results}
        assert "v1" in result_ids
        assert "v3" in result_ids
        assert "v2" in result_ids

        # Verify distances approximately
        dist_v1 = euclidean_distance(query_vec.data, populated_m_tree_euclidean.search_knn(Vector("x", [1.0, 1.0]), 1)[0].vector.data)
        dist_v3 = euclidean_distance(query_vec.data, populated_m_tree_euclidean.search_knn(Vector("x", [1.1, 1.0]), 1)[0].vector.data)
        dist_v2 = euclidean_distance(query_vec.data, populated_m_tree_euclidean.search_knn(Vector("x", [2.0, 2.0]), 1)[0].vector.data)

        # Retrieve the exact distances from the results and compare
        result_dist_v1 = next(r.distance for r in results if r.vector.id == "v1")
        result_dist_v3 = next(r.distance for r in results if r.vector.id == "v3")
        result_dist_v2 = next(r.distance for r in results if r.vector.id == "v2")

        assert pytest.approx(result_dist_v1, abs=1e-4) == euclidean_distance(query_vec.data, np.array([1.0, 1.0]))
        assert pytest.approx(result_dist_v3, abs=1e-4) == euclidean_distance(query_vec.data, np.array([1.1, 1.0]))
        assert pytest.approx(result_dist_v2, abs=1e-4) == euclidean_distance(query_vec.data, np.array([2.0, 2.0]))

        # Ensure results are sorted by distance
        assert results[0].distance <= results[1].distance <= results[2].distance

    def test_search_knn_cosine_correctness(self, populated_m_tree_cosine):
        # Query near c1 ([1,0]) and c3 ([1,1]) which is also close to c5 ([0.5,0.5])
        query_vec = Vector("q_near_c1_c3", [1.0, 0.1]) 
        results = populated_m_tree_cosine.search_knn(query_vec, k=3)

        assert len(results) == 3

        # Expected based on 1 - cosine_similarity
        # c1: [1.0, 0.0] -> cos_sim([1,0.1],[1,0]) ~ 0.995 -> dist ~ 0.005
        # c3: [1.0, 1.0] -> cos_sim([1,0.1],[1,1]) ~ 0.714 -> dist ~ 0.286
        # c5: [0.5, 0.5] -> cos_sim([1,0.1],[0.5,0.5]) ~ 0.714 -> dist ~ 0.286
        # c2: [0.0, 1.0] -> cos_sim([1,0.1],[0,1]) ~ 0.0995 -> dist ~ 0.9005

        result_ids = {r.vector.id for r in results}
        assert "c1" in result_ids
        assert "c3" in result_ids
        assert "c5" in result_ids # c5 and c3 are equally similar in angle

        # Verify distances approximately
        expected_dist_c1 = 1 - cosine_similarity(query_vec.data, np.array([1.0, 0.0]))
        expected_dist_c3 = 1 - cosine_similarity(query_vec.data, np.array([1.0, 1.0]))
        expected_dist_c5 = 1 - cosine_similarity(query_vec.data, np.array([0.5, 0.5]))

        result_dist_c1 = next(r.distance for r in results if r.vector.id == "c1")
        result_dist_c3 = next(r.distance for r in results if r.vector.id == "c3")
        result_dist_c5 = next(r.distance for r in results if r.vector.id == "c5")

        assert pytest.approx(result_dist_c1, abs=1e-4) == expected_dist_c1
        assert pytest.approx(result_dist_c3, abs=1e-4) == expected_dist_c3
        assert pytest.approx(result_dist_c5, abs=1e-4) == expected_dist_c5

        assert results[0].distance <= results[1].distance <= results[2].distance


    def test_search_range_empty_tree(self, empty_m_tree_euclidean):
        query_vec = Vector("q", [0.0, 0.0])
        results = empty_m_tree_euclidean.search_range(query_vec, radius=1.0)
        assert len(results) == 0

    def test_search_range_euclidean_correctness(self, populated_m_tree_euclidean):
        query_vec = Vector("q_range", [1.5, 1.5])
        radius = 1.0 # Should include v1, v2, v3, v8, v11, v14, v15
        results = populated_m_tree_euclidean.search_range(query_vec, radius)
        
        # Calculate expected based on Euclidean
        expected_ids = set()
        for vec_data in [
            ([1.0, 1.0], "v1"), ([2.0, 2.0], "v2"), ([1.1, 1.0], "v3"), ([0.5, 0.5], "v8"),
            ([2.5, 2.5], "v11"), ([3.0, 3.0], "v14"), ([4.0, 4.0], "v15")
        ]:
            if euclidean_distance(query_vec.data, np.array(vec_data[0])) <= radius:
                expected_ids.add(vec_data[1])

        result_ids = {r.vector.id for r in results}
        assert result_ids == expected_ids
        assert len(results) == len(expected_ids)

        # Test with a radius that should yield no results
        query_vec_no_results = Vector("q_none", [100.0, 100.0])
        results_no_results = populated_m_tree_euclidean.search_range(query_vec_no_results, radius=0.1)
        assert len(results_no_results) == 0

    def test_search_range_cosine_correctness(self, populated_m_tree_cosine):
        query_vec = Vector("q_range_cosine", [0.9, 0.2]) # Close to c1, c3, c5
        radius = 0.3 # Corresponds to cosine similarity approx 0.7
        results = populated_m_tree_cosine.search_range(query_vec, radius)

        expected_ids = set()
        vectors_in_tree = [
            (np.array([1.0, 0.0]), "c1"),
            (np.array([0.0, 1.0]), "c2"),
            (np.array([1.0, 1.0]), "c3"),
            (np.array([-1.0, 0.0]), "c4"),
            (np.array([0.5, 0.5]), "c5"),
            (np.array([0.0, -1.0]), "c6")
        ]

        for vec_data_np, vec_id in vectors_in_tree:
            if (1 - cosine_similarity(query_vec.data, vec_data_np)) <= radius:
                expected_ids.add(vec_id)
        
        result_ids = {r.vector.id for r in results}
        assert result_ids == expected_ids
        assert len(results) == len(expected_ids)

        # Test with a radius that should yield no results
        query_vec_no_results = Vector("q_none", [-0.9, 0.1]) # Far from most positive vectors
        results_no_results = populated_m_tree_cosine.search_range(query_vec_no_results, radius=0.1)
        
        expected_ids_no_results = set()
        for vec_data_np, vec_id in vectors_in_tree:
            if (1 - cosine_similarity(query_vec_no_results.data, vec_data_np)) <= 0.1:
                expected_ids_no_results.add(vec_id)
        
        assert len(results_no_results) == 0 or len(results_no_results) == len(expected_ids_no_results) # should be 0 if no match
        # Specifically for this query_vec_no_results and radius 0.1, it's very likely 0 results.
        if "c4" in result_ids and pytest.approx(1 - cosine_similarity(query_vec_no_results.data, np.array([-1.0, 0.0])), abs=1e-4) <= 0.1:
            pass # c4 might be close depending on random small values.
        else:
            assert len(results_no_results) == 0

    def test_vector_update_or_replacement(self, populated_m_tree_euclidean):
        # Insert a vector with an existing ID, check if it updates or adds a new entry
        # Current M-Tree implementation adds a new entry if ID exists, unless specific handling is added.
        # For this test, we assume it adds a new entry if ID is repeated.
        initial_size = populated_m_tree_euclidean.size()
        existing_vec_id = "v1"
        updated_vec = Vector(existing_vec_id, [100.0, 100.0], metadata={"updated": True})
        
        populated_m_tree_euclidean.insert(updated_vec)
        assert populated_m_tree_euclidean.size() == initial_size + 1 # Expecting an addition, not an update

        # If you were to implement update logic, this test would change.
        # For now, it tests the current behavior (inserting a new entry with duplicate ID).
        # To truly test "update", you'd need a dedicated `update` method or modify `insert` to handle it.
        
        # Verify both original and "updated" vector can be found if no update logic
        results_v1_old = populated_m_tree_euclidean.search_knn(Vector("query_v1", [1.0, 1.0]), k=1)
        results_v1_new = populated_m_tree_euclidean.search_knn(Vector("query_v1_new", [100.0, 100.0]), k=1)
        assert results_v1_old[0].vector.id == "v1"
        assert results_v1_new[0].vector.id == "v1"
        assert results_v1_new[0].vector.metadata == {"updated": True} # Verify it's the new one.


    def test_insert_vector_different_dimensions_raises_error(self, empty_m_tree_euclidean):
        vec1 = Vector("v1", [1.0, 2.0])
        empty_m_tree_euclidean.insert(vec1)

        vec2_bad_dim = Vector("v2", [1.0, 2.0, 3.0]) # Different dimension
        with pytest.raises(ValueError, match="Dimensions must match"):
            empty_m_tree_euclidean.insert(vec2_bad_dim)

    def test_search_query_vector_different_dimensions_raises_error(self, populated_m_tree_euclidean):
        query_vec_bad_dim = Vector("q", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Query vector dimension must match tree's vector dimension"):
            populated_m_tree_euclidean.search_knn(query_vec_bad_dim, k=1)
        
        with pytest.raises(ValueError, match="Query vector dimension must match tree's vector dimension"):
            populated_m_tree_euclidean.search_range(query_vec_bad_dim, radius=1.0)