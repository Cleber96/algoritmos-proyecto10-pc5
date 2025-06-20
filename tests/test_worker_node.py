# tests/test_worker_node.py
import pytest
import requests
import subprocess
import time
import os
import sys
import json

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.models import Vector, SearchResult
from src.common.utils import log_info, log_error
from src.simulation.data_generator import generate_random_vectors

WORKER_PORT = 6001 # Usar un puerto diferente para evitar conflictos con demos
WORKER_ID = "test_worker_1"
WORKER_URL = f"http://localhost:{WORKER_PORT}"
VECTOR_DIM = 32

@pytest.fixture(scope="module")
def worker_server():
    """Fixture para iniciar y detener el servidor Worker para las pruebas."""
    log_info(f"--- Setting up Worker server for tests on {WORKER_URL} ---")
    
    # Iniciar el proceso del worker
    # Asegúrate de que 'python' apunta al intérprete de tu entorno virtual.
    # cwd es crucial para que el script 'app.py' encuentre sus módulos.
    worker_proc = subprocess.Popen(
        [sys.executable, 'src/worker_node/app.py', '--id', WORKER_ID, '--port', str(WORKER_PORT)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
        stdout=subprocess.PIPE, # Capturar salida para depuración si es necesario
        stderr=subprocess.PIPE
    )
    
    # Esperar a que el servidor esté listo
    timeout = 10
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"{WORKER_URL}/status", timeout=1)
            if response.status_code == 200:
                log_info(f"Worker server running on {WORKER_URL}")
                break
        except requests.exceptions.ConnectionError:
            pass
        if time.time() - start_time > timeout:
            worker_proc.terminate()
            stdout, stderr = worker_proc.communicate()
            log_error(f"Worker did not start in time. Stdout: {stdout.decode()}, Stderr: {stderr.decode()}")
            pytest.fail(f"Worker server failed to start on {WORKER_URL}")
        time.sleep(0.5)

    yield # Ejecutar las pruebas

    # Teardown: detener el proceso del worker
    log_info(f"--- Tearing down Worker server {WORKER_URL} ---")
    if worker_proc.poll() is None: # Si el proceso sigue corriendo
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_proc.kill()
            log_error(f"Worker process {worker_proc.pid} did not terminate gracefully, killed.")
    log_info("Worker server stopped.")

class TestWorkerNodeAPI:
    def test_status_endpoint(self, worker_server):
        response = requests.get(f"{WORKER_URL}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == WORKER_ID
        assert "m_tree_size" in data
        assert "distance_metric" in data

    def test_insert_vector(self, worker_server):
        test_vector = Vector("test_vec_1", [0.1] * VECTOR_DIM, metadata={"source": "test"})
        response = requests.post(f"{WORKER_URL}/insert_vector", json=test_vector.to_dict())
        assert response.status_code == 200
        assert response.json()["status"] == "Vector inserted"
        
        # Verify size increased
        status_response = requests.get(f"{WORKER_URL}/status").json()
        assert status_response["m_tree_size"] == 1 # Assuming a clean slate for each test, or manage state

    def test_insert_multiple_vectors(self, worker_server):
        # Clear tree first if it retains state across tests
        # (Not strictly unit test practice but necessary for integration tests with shared state)
        # Assuming you have a /clear endpoint or restart server per test.
        # For simplicity, we'll just insert more.
        initial_size_response = requests.get(f"{WORKER_URL}/status").json()
        initial_size = initial_size_response.get("m_tree_size", 0)

        vectors = generate_random_vectors(10, VECTOR_DIM, start_id=100)
        for vec in vectors:
            response = requests.post(f"{WORKER_URL}/insert_vector", json=vec.to_dict())
            assert response.status_code == 200
            assert response.json()["status"] == "Vector inserted"
        
        status_response = requests.get(f"{WORKER_URL}/status").json()
        assert status_response["m_tree_size"] == initial_size + 10

    def test_insert_invalid_vector_data(self, worker_server):
        # Missing 'id'
        response = requests.post(f"{WORKER_URL}/insert_vector", json={"vector": [1.0] * VECTOR_DIM})
        assert response.status_code == 400
        assert "error" in response.json()

        # Invalid 'vector' type
        response = requests.post(f"{WORKER_URL}/insert_vector", json={"id": "bad_id", "vector": "not_a_list"})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_search_knn(self, worker_server):
        # Ensure there are vectors in the tree for search
        vec1 = Vector("s_vec_1", [1.0] * VECTOR_DIM)
        vec2 = Vector("s_vec_2", [1.1] * VECTOR_DIM)
        vec3 = Vector("s_vec_3", [2.0] * VECTOR_DIM)
        requests.post(f"{WORKER_URL}/insert_vector", json=vec1.to_dict())
        requests.post(f"{WORKER_URL}/insert_vector", json=vec2.to_dict())
        requests.post(f"{WORKER_URL}/insert_vector", json=vec3.to_dict())
        
        query_vec = Vector("query_k_1", [1.05] * VECTOR_DIM)
        response = requests.post(
            f"{WORKER_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": 2}
        )
        assert response.status_code == 200
        results = response.json().get("results", [])
        assert len(results) == 2
        # Check if expected vectors are in results and sorted by distance
        result_ids = {r["vector_id"] for r in results}
        assert "s_vec_1" in result_ids
        assert "s_vec_2" in result_ids
        assert results[0]["distance"] <= results[1]["distance"]

    def test_search_knn_no_results(self, worker_server):
        # Clear worker or ensure it's empty if possible. For demo, assume
        # test_insert_multiple_vectors has already run.
        # Query far away from any existing vectors
        query_vec = Vector("query_far", [100.0] * VECTOR_DIM)
        response = requests.post(
            f"{WORKER_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": 1}
        )
        assert response.status_code == 200
        results = response.json().get("results", [])
        # If the tree has elements, it should return 1 result (the closest one)
        # If the tree is empty or the query is bad, it might return 0.
        # For a general test, we expect at least 1 result if the tree is populated.
        status_response = requests.get(f"{WORKER_URL}/status").json()
        if status_response["m_tree_size"] > 0:
            assert len(results) == 1
        else:
            assert len(results) == 0


    def test_search_knn_invalid_k(self, worker_server):
        query_vec = Vector("query", [0.0] * VECTOR_DIM)
        # k=0
        response = requests.post(
            f"{WORKER_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": 0}
        )
        assert response.status_code == 400
        assert "error" in response.json()

        # k is not an integer
        response = requests.post(
            f"{WORKER_URL}/search/knn",
            json={"query_vector": query_vec.to_dict(), "k": "abc"}
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_search_range(self, worker_server):
        # Ensure vectors are present
        vec1 = Vector("r_vec_1", [1.0] * VECTOR_DIM)
        vec2 = Vector("r_vec_2", [1.1] * VECTOR_DIM)
        vec3 = Vector("r_vec_3", [5.0] * VECTOR_DIM)
        requests.post(f"{WORKER_URL}/insert_vector", json=vec1.to_dict())
        requests.post(f"{WORKER_URL}/insert_vector", json=vec2.to_dict())
        requests.post(f"{WORKER_URL}/insert_vector", json=vec3.to_dict())

        query_vec = Vector("query_r_1", [1.05] * VECTOR_DIM)
        # A small radius should only find vec1 and vec2
        response = requests.post(
            f"{WORKER_URL}/search/range",
            json={"query_vector": query_vec.to_dict(), "radius": 0.5}
        )
        assert response.status_code == 200
        results = response.json().get("results", [])
        result_ids = {r["vector_id"] for r in results}
        
        # Depending on exact distances, this might be 1 or 2 vectors.
        # For [1.05]*dim, vec1 ([1.0]*dim) and vec2 ([1.1]*dim) are very close.
        # Euclidean distance between [1.05]*dim and [1.0]*dim is sqrt(dim * 0.05^2)
        # sqrt(32 * 0.0025) = sqrt(0.08) approx 0.28. So radius 0.5 includes both.
        assert "r_vec_1" in result_ids
        assert "r_vec_2" in result_ids
        assert len(results) >= 2 # Should find at least vec1 and vec2

        # A radius that yields no results
        response_no_results = requests.post(
            f"{WORKER_URL}/search/range",
            json={"query_vector": Vector("query_far_r", [100.0] * VECTOR_DIM).to_dict(), "radius": 0.1}
        )
        assert response_no_results.status_code == 200
        assert len(response_no_results.json().get("results", [])) == 0

    def test_search_range_invalid_radius(self, worker_server):
        query_vec = Vector("query", [0.0] * VECTOR_DIM)
        # radius <= 0
        response = requests.post(
            f"{WORKER_URL}/search/range",
            json={"query_vector": query_vec.to_dict(), "radius": -1.0}
        )
        assert response.status_code == 400
        assert "error" in response.json()