# tests/test_orchestrator.py
import pytest
import requests
import numpy as np
import subprocess
import time
import os
import sys
import json
from typing import List, Dict, Any

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.models import Vector, SearchResult
from src.common.utils import log_info, log_error, log_warning
from src.simulation.data_generator import generate_random_vectors

ORCHESTRATOR_PORT = 7000 # Puerto diferente para el orquestador de pruebas
ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"
WORKER_BASE_PORT = 7001 # Puertos base para workers de pruebas

NUM_WORKERS_FOR_TESTS = 3
VECTOR_DIM = 16 # Dimensión más pequeña para pruebas rápidas
TEST_TIMEOUT_SEC = 15 # Timeout general para operaciones de red

@pytest.fixture(scope="module")
def distributed_system():
    """Fixture para iniciar y detener el sistema distribuido (Orquestador + Workers)."""
    log_info("\n--- Setting up distributed system for Orchestrator tests ---")
    
    orchestrator_proc = None
    worker_procs: List[subprocess.Popen] = []
    worker_urls: List[str] = []

    try:
        # 1. Iniciar Orquestador
        log_info(f"Starting Orchestrator on {ORCHESTRATOR_URL}...")
        orchestrator_proc = subprocess.Popen(
            [sys.executable, 'src/orchestrator/app.py', '--port', str(ORCHESTRATOR_PORT)],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(1) # Give it a moment to start
        
        # Wait for orchestrator to be ready
        _wait_for_server(ORCHESTRATOR_URL, "Orchestrator")

        # 2. Iniciar Workers
        for i in range(NUM_WORKERS_FOR_TESTS):
            worker_id = f"test_worker_{i+1}"
            worker_port = WORKER_BASE_PORT + i
            worker_url = f"http://localhost:{worker_port}"
            worker_proc = subprocess.Popen(
                [sys.executable, 'src/worker_node/app.py', '--id', worker_id, '--port', str(worker_port)],
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            worker_procs.append(worker_proc)
            worker_urls.append(worker_url)
            log_info(f"Worker '{worker_id}' started on {worker_url}.")
        
        time.sleep(1) # Give workers a moment to start
        
        # Wait for workers to be ready
        for url in worker_urls:
            _wait_for_server(url, f"Worker {url}")

        # 3. Registrar Workers con el Orquestador
        log_info("Registering workers with Orchestrator...")
        for i in range(NUM_WORKERS_FOR_TESTS):
            worker_id = f"test_worker_{i+1}"
            worker_url = worker_urls[i]
            response = requests.post(
                f"{ORCHESTRATOR_URL}/register_worker",
                json={"node_id": worker_id, "node_url": worker_url},
                timeout=TEST_TIMEOUT_SEC
            )
            response.raise_for_status()
            assert response.json().get("status") == "Worker registered"
            log_info(f"Registered worker {worker_id}")

        time.sleep(1) # Short delay after registration

        yield orchestrator_proc, worker_procs, worker_urls # Yield control to tests

    finally:
        log_info("\n--- Tearing down distributed system ---")
        # Terminar workers primero
        for proc in worker_procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    log_error(f"Worker process {proc.pid} did not terminate gracefully, killed.")
        
        # Terminar orquestador
        if orchestrator_proc and orchestrator_proc.poll() is None:
            orchestrator_proc.terminate()
            try:
                orchestrator_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                orchestrator_proc.kill()
                log_error(f"Orchestrator process {orchestrator_proc.pid} did not terminate gracefully, killed.")
        log_info("Distributed system stopped.")

def _wait_for_server(url: str, server_name: str, timeout: int = 10):
    """Helper function to wait for a server to become available."""
    start_time = time.time()
    while True:
        try:
            requests.get(f"{url}/status", timeout=1).raise_for_status()
            log_info(f"{server_name} is up.")
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            pass # Server not ready or returned error
        if time.time() - start_time > timeout:
            pytest.fail(f"{server_name} failed to start or respond at {url} within {timeout} seconds.")
        time.sleep(0.5)

class TestOrchestratorAPI:
    def test_orchestrator_status(self, distributed_system):
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=TEST_TIMEOUT_SEC)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["active_workers_count"] == NUM_WORKERS_FOR_TESTS
        assert len(data["worker_details"]) == NUM_WORKERS_FOR_TESTS

    def test_register_duplicate_worker(self, distributed_system):
        # Try to register an already registered worker
        worker_id = "test_worker_1"
        worker_url = f"http://localhost:{WORKER_BASE_PORT}"
        response = requests.post(
            f"{ORCHESTRATOR_URL}/register_worker",
            json={"node_id": worker_id, "node_url": worker_url},
            timeout=TEST_TIMEOUT_SEC
        )
        assert response.status_code == 409 # Conflict status code
        assert "error" in response.json()
        assert "already registered" in response.json()["error"]

    def test_unregister_worker(self, distributed_system):
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        
        worker_to_unregister = "test_worker_1"
        response = requests.post(
            f"{ORCHESTRATOR_URL}/unregister_worker",
            json={"node_id": worker_to_unregister},
            timeout=TEST_TIMEOUT_SEC
        )
        assert response.status_code == 200
        assert response.json()["status"] == "Worker unregistered"

        # Verify worker is no longer in status
        status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=TEST_TIMEOUT_SEC).json()
        assert status_response["active_workers_count"] == NUM_WORKERS_FOR_TESTS - 1
        assert worker_to_unregister not in [w["node_id"] for w in status_response["worker_details"]]
        
        # Re-register for subsequent tests in the same fixture run
        # This is important if fixture scope is module and tests depend on full setup
        requests.post(
            f"{ORCHESTRATOR_URL}/register_worker",
            json={"node_id": worker_to_unregister, "node_url": f"http://localhost:{WORKER_BASE_PORT}"},
            timeout=TEST_TIMEOUT_SEC
        ).raise_for_status()
        time.sleep(0.5)

    def test_unregister_non_existent_worker(self, distributed_system):
        response = requests.post(
            f"{ORCHESTRATOR_URL}/unregister_worker",
            json={"node_id": "non_existent_worker"},
            timeout=TEST_TIMEOUT_SEC
        )
        assert response.status_code == 404
        assert "error" in response.json()
        assert "not found" in response.json()["error"]

    def test_insert_vector_distributed(self, distributed_system):
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        
        # Get initial worker sizes
        initial_worker_sizes = {}
        for worker_url in worker_urls:
            status_res = requests.get(f"{worker_url}/status", timeout=TEST_TIMEOUT_SEC).json()
            initial_worker_sizes[status_res["id"]] = status_res["m_tree_size"]

        num_vectors_to_insert = 10
        vectors = generate_random_vectors(num_vectors_to_insert, VECTOR_DIM, start_id=1000)

        for vec in vectors:
            response = requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec.to_dict(), timeout=TEST_TIMEOUT_SEC)
            assert response.status_code == 200
            assert response.json()["status"] == "Vector distributed and inserted"
        
        time.sleep(1) # Give workers time to process

        # Verify vectors are distributed
        final_worker_sizes = {}
        total_inserted = 0
        for worker_url in worker_urls:
            status_res = requests.get(f"{worker_url}/status", timeout=TEST_TIMEOUT_SEC).json()
            final_worker_sizes[status_res["id"]] = status_res["m_tree_size"]
            total_inserted += (final_worker_sizes[status_res["id"]] - initial_worker_sizes[status_res["id"]])
        
        assert total_inserted == num_vectors_to_insert
        
        # Check for some distribution across multiple workers (not all on one)
        # This is probabilistic, so check if at least 2 workers received data.
        workers_with_new_data = [w_id for w_id, size in final_worker_sizes.items() if size > initial_worker_sizes[w_id]]
        assert len(workers_with_new_data) > 1 # Expecting distribution

    def test_search_knn_distributed(self, distributed_system):
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        
        # Ensure some data exists for search (test_insert_vector_distributed should have run)
        # Insert a known vector to test search accuracy
        known_vec = Vector("known_vec_1", [0.5] * VECTOR_DIM, metadata={"tag": "test"})
        requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=known_vec.to_dict(), timeout=TEST_TIMEOUT_SEC).raise_for_status()
        time.sleep(0.5)

        query_vec = Vector("query_knn_dist", [0.51] * VECTOR_DIM)
        k_val = 3
        response = requests.post(
            f"{ORCHESTRATOR_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": k_val},
            timeout=TEST_TIMEOUT_SEC + 5 # Give more time for distributed search
        )
        assert response.status_code == 200
        results_raw = response.json().get("results", [])
        
        # Check if the known vector is among the results
        result_ids = {r["vector_id"] for r in results_raw}
        assert "known_vec_1" in result_ids
        
        # Check that k results are returned (unless fewer vectors exist in total)
        # Need to know the total size of all workers to assert exact k.
        # For simplicity, assert it returns *up to* k results and is sorted.
        total_vectors_in_system = sum(
            requests.get(f"{url}/status", timeout=TEST_TIMEOUT_SEC).json()["m_tree_size"] 
            for url in worker_urls
        )
        assert len(results_raw) <= k_val and len(results_raw) <= total_vectors_in_system
        
        # Check sorting by distance
        for i in range(len(results_raw) - 1):
            assert results_raw[i]["distance"] <= results_raw[i+1]["distance"]

    def test_search_range_distributed(self, distributed_system):
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        
        # Insert known vectors for range search
        vec_in_range1 = Vector("range_vec_A", [0.1] * VECTOR_DIM)
        vec_in_range2 = Vector("range_vec_B", [0.2] * VECTOR_DIM)
        vec_out_of_range = Vector("range_vec_C", [10.0] * VECTOR_DIM)

        requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec_in_range1.to_dict(), timeout=TEST_TIMEOUT_SEC).raise_for_status()
        requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec_in_range2.to_dict(), timeout=TEST_TIMEOUT_SEC).raise_for_status()
        requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec_out_of_range.to_dict(), timeout=TEST_TIMEOUT_SEC).raise_for_status()
        time.sleep(0.5)

        query_vec = Vector("query_range_dist", [0.15] * VECTOR_DIM)
        search_radius = 0.1 # This radius should include A and B if they are close enough

        # Expected distance calculation (Euclidean for simplicity)
        dist_A = np.linalg.norm(query_vec.data - np.array([0.1]*VECTOR_DIM)) # sqrt(dim * 0.05^2)
        dist_B = np.linalg.norm(query_vec.data - np.array([0.2]*VECTOR_DIM)) # sqrt(dim * 0.05^2)
        
        response = requests.post(
            f"{ORCHESTRATOR_URL}/search/range",
            json={"query_vector": query_vec.to_dict(), "radius": search_radius},
            timeout=TEST_TIMEOUT_SEC + 5
        )
        assert response.status_code == 200
        results_raw = response.json().get("results", [])
        
        result_ids = {r["vector_id"] for r in results_raw}
        
        # Given VECTOR_DIM=16, radius=0.1:
        # dist([0.15]*16, [0.1]*16) = sqrt(16 * (0.05)^2) = sqrt(16 * 0.0025) = sqrt(0.04) = 0.2
        # dist([0.15]*16, [0.2]*16) = sqrt(16 * (0.05)^2) = sqrt(16 * 0.0025) = sqrt(0.04) = 0.2
        # So with radius 0.1, neither A nor B will be found. This needs a larger radius.
        # Let's adjust expected radius to get results, or query to fit radius.
        
        # Recalculating based on a more sensible radius
        search_radius = 0.3 # This should include A and B (distances are 0.2)
        response = requests.post(
            f"{ORCHESTRATOR_URL}/search/range",
            json={"query_vector": query_vec.to_dict(), "radius": search_radius},
            timeout=TEST_TIMEOUT_SEC + 5
        )
        assert response.status_code == 200
        results_raw = response.json().get("results", [])
        result_ids = {r["vector_id"] for r in results_raw}

        assert "range_vec_A" in result_ids
        assert "range_vec_B" in result_ids
        assert "range_vec_C" not in result_ids # Should be out of range
        assert len(results_raw) == 2 # Expecting exactly 2 results

        # Test with no workers registered (temporary unregister all)
        orchestrator_proc, worker_procs, worker_urls = distributed_system
        for worker_id in [f"test_worker_{i+1}" for i in range(NUM_WORKERS_FOR_TESTS)]:
            requests.post(f"{ORCHESTRATOR_URL}/unregister_worker", json={"node_id": worker_id}, timeout=TEST_TIMEOUT_SEC).raise_for_status()
        time.sleep(0.5) # Give orchestrator time to update its state

        response_no_workers = requests.post(
            f"{ORCHESTRATOR_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": 1},
            timeout=TEST_TIMEOUT_SEC
        )
        assert response_no_workers.status_code == 503 # Service Unavailable
        assert "No active workers" in response_no_workers.json()["error"]

        # Re-register workers to restore system state for other tests if fixture scope is module
        for i in range(NUM_WORKERS_FOR_TESTS):
            worker_id = f"test_worker_{i+1}"
            worker_url = worker_urls[i]
            requests.post(f"{ORCHESTRATOR_URL}/register_worker", json={"node_id": worker_id, "node_url": worker_url}, timeout=TEST_TIMEOUT_SEC).raise_for_status()
        time.sleep(1) # Give orchestrator time to rebuild ring