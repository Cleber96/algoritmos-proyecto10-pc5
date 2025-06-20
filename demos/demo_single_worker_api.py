# demos/demo_single_worker_api.py
import sys
import os
import time
import requests
import subprocess
import json
import numpy as np

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.models import Vector, SearchResult
from src.common.utils import log_info, log_error
from src.simulation.data_generator import generate_random_vectors # Para generar datos de prueba

WORKER_PORT = 5001
WORKER_URL = f"http://localhost:{WORKER_PORT}"
WORKER_ID = "single_worker_demo"
VECTOR_DIM = 64
NUM_VECTORS = 10
K_NN = 3

def start_worker_process():
    """Inicia el proceso del nodo worker en segundo plano."""
    log_info(f"Iniciando Worker '{WORKER_ID}' en {WORKER_URL}...")
    # Usar Popen para ejecutar la app de Flask/FastAPI en un proceso separado
    # Es crucial que el comando 'python' sea el que está en tu entorno virtual.
    # stdout=subprocess.PIPE, stderr=subprocess.PIPE puede ser útil para depurar
    worker_proc = subprocess.Popen(
        ['python', 'src/worker_node/app.py', '--id', WORKER_ID, '--port', str(WORKER_PORT)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Ruta base del proyecto
    )
    time.sleep(3) # Espera un poco para que el servidor se inicie
    log_info(f"Worker iniciado con PID: {worker_proc.pid}")
    return worker_proc

def stop_worker_process(worker_proc):
    """Detiene el proceso del nodo worker."""
    if worker_proc.poll() is None: # Si el proceso sigue corriendo
        log_info(f"Deteniendo Worker con PID: {worker_proc.pid}")
        worker_proc.terminate()
        worker_proc.wait(timeout=5) # Espera a que termine
        if worker_proc.poll() is None: # Si no terminó, mátalo
            worker_proc.kill()
            log_warning(f"Worker con PID: {worker_proc.pid} fue forzado a terminar.")
    log_info("Worker detenido.")

def run_single_worker_api_demo():
    """
    Demostración de la API de un único nodo Worker:
    insertar vectores y realizar búsquedas k-NN.
    """
    log_info("--- Iniciando demostración de la API de un solo Worker ---")

    worker_process = None
    try:
        worker_process = start_worker_process()

        # 1. Verificar estado del Worker
        log_info("\n--- Verificando estado del Worker ---")
        try:
            response = requests.get(f"{WORKER_URL}/status")
            response.raise_for_status() # Lanza excepción si el código de estado es un error
            log_info(f"Estado del Worker: {response.json()}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al conectar con el Worker: {e}")
            log_error("Asegúrate de que el Worker Node esté corriendo correctamente.")
            return

        # 2. Insertar vectores
        log_info(f"\n--- Insertando {NUM_VECTORS} vectores en el Worker ---")
        vectors_to_insert = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)
        
        for i, vec in enumerate(vectors_to_insert):
            try:
                response = requests.post(f"{WORKER_URL}/insert_vector", json=vec.to_dict())
                response.raise_for_status()
                log_info(f"Vector {vec.id} insertado. Respuesta: {response.json()}")
            except requests.exceptions.RequestException as e:
                log_error(f"Error al insertar vector {vec.id}: {e}")
                break
            time.sleep(0.1) # Pequeña pausa para no sobrecargar si hay muchos vectores

        # 3. Realizar búsqueda k-NN
        log_info(f"\n--- Realizando búsqueda k-NN (k={K_NN}) ---")
        query_vector = generate_random_vectors(1, VECTOR_DIM, start_id=1000)[0]
        log_info(f"Vector de consulta: {query_vector.data.tolist()[:5]}...")

        try:
            response = requests.post(
                f"{WORKER_URL}/search/knn",
                json={"query_vector": query_vector.to_dict(), "k": K_NN}
            )
            response.raise_for_status()
            search_results_raw = response.json().get("results", [])
            log_info(f"Resultados de búsqueda ({len(search_results_raw)}):")
            for res_data in search_results_raw:
                res = SearchResult.from_dict(res_data)
                log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Metadatos: {res.vector.metadata}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al realizar búsqueda k-NN: {e}")

        # 4. Realizar búsqueda por rango (opcional)
        log_info(f"\n--- Realizando búsqueda por rango (radio=1.0) ---")
        query_vector_range = generate_random_vectors(1, VECTOR_DIM, start_id=2000)[0]
        search_radius = 1.0
        log_info(f"Vector de consulta: {query_vector_range.data.tolist()[:5]}...")
        
        try:
            response = requests.post(
                f"{WORKER_URL}/search/range",
                json={"query_vector": query_vector_range.to_dict(), "radius": search_radius}
            )
            response.raise_for_status()
            search_results_range_raw = response.json().get("results", [])
            log_info(f"Resultados de búsqueda por rango ({len(search_results_range_raw)}):")
            for res_data in search_results_range_raw:
                res = SearchResult.from_dict(res_data)
                log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Metadatos: {res.vector.metadata}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al realizar búsqueda por rango: {e}")


    finally:
        if worker_process:
            stop_worker_process(worker_process)
        log_info("--- Demostración de la API de un solo Worker completada ---")

if __name__ == "__main__":
    run_single_worker_api_demo()