# demos/demo_orchestrator_interaction.py
import sys
import os
import time
import requests
import subprocess
import json
import numpy as np
from typing import List, Dict, Any

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.models import Vector, SearchResult
from src.common.utils import log_info, log_error, log_warning
from src.simulation.data_generator import generate_random_vectors

ORCHESTRATOR_PORT = 5000
WORKER_BASE_PORT = 5001
ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"
NUM_WORKERS = 3 # Puedes cambiar este número para probar diferentes configuraciones
VECTOR_DIM = 64
TOTAL_VECTORS_TO_INSERT = 30
K_NN_SEARCH = 3

# Retraso para asegurar que los servicios estén listos
STARTUP_DELAY = 3 # segundos

def start_system(num_workers: int) -> tuple[subprocess.Popen, List[subprocess.Popen]]:
    """Inicia el orquestador y los workers, y registra los workers con el orquestador."""
    log_info(f"Iniciando sistema con {num_workers} workers...")
    
    orchestrator_proc = subprocess.Popen(
        ['python', 'src/orchestrator/app.py', '--port', str(ORCHESTRATOR_PORT)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    log_info(f"Orquestador iniciado (PID: {orchestrator_proc.pid}).")
    time.sleep(STARTUP_DELAY) # Dale tiempo para iniciar

    worker_procs: List[subprocess.Popen] = []
    worker_urls: List[str] = []

    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        worker_port = WORKER_BASE_PORT + i
        worker_url = f"http://localhost:{worker_port}"
        
        proc = subprocess.Popen(
            ['python', 'src/worker_node/app.py', '--id', worker_id, '--port', str(worker_port)],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        worker_procs.append(proc)
        worker_urls.append(worker_url)
        log_info(f"Worker '{worker_id}' iniciado (PID: {proc.pid}) en {worker_url}.")

    time.sleep(STARTUP_DELAY) # Dale tiempo a los workers para iniciar

    # Registrar workers con el orquestador
    log_info("\n--- Registrando workers con el Orquestador ---")
    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        worker_url = worker_urls[i]
        try:
            response = requests.post(
                f"{ORCHESTRATOR_URL}/register_worker", 
                json={"node_id": worker_id, "node_url": worker_url},
                timeout=5
            )
            response.raise_for_status()
            log_info(f"Worker '{worker_id}' registrado: {response.json().get('status')}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al registrar worker '{worker_id}': {e}")
            raise # Re-lanzar para detener la demo si el registro falla

    time.sleep(1) # Pequeña pausa después del registro
    return orchestrator_proc, worker_procs

def stop_system(orchestrator_proc: subprocess.Popen, worker_procs: List[subprocess.Popen]):
    """Detiene todos los procesos del sistema."""
    log_info("\n--- Deteniendo el sistema distribuido ---")
    
    # Terminar workers primero
    for proc in worker_procs:
        if proc.poll() is None: # Si el proceso sigue corriendo
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                log_warning(f"Worker PID {proc.pid} fue forzado a terminar.")
        log_info(f"Worker PID {proc.pid} detenido.")

    # Terminar orquestador
    if orchestrator_proc.poll() is None:
        orchestrator_proc.terminate()
        try:
            orchestrator_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            orchestrator_proc.kill()
            log_warning(f"Orquestador PID {orchestrator_proc.pid} fue forzado a terminar.")
    log_info(f"Orquestador PID {orchestrator_proc.pid} detenido.")
    
    log_info("Sistema detenido completamente.")


def run_orchestrator_interaction_demo():
    """
    Demostración de la interacción entre el Orquestador y múltiples Workers.
    """
    log_info("--- Iniciando demostración de interacción Orquestador-Workers ---")

    orchestrator_proc, worker_procs = None, []
    try:
        orchestrator_proc, worker_procs = start_system(NUM_WORKERS)
        
        # 1. Verificar estado del sistema
        log_info("\n--- Verificando estado del Orquestador ---")
        try:
            status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=5)
            status_response.raise_for_status()
            log_info(f"Estado del Orquestador: {status_response.json()}")
            active_workers = status_response.json().get('active_workers_count', 0)
            if active_workers != NUM_WORKERS:
                log_warning(f"Esperábamos {NUM_WORKERS} workers activos, pero el orquestador reporta {active_workers}.")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al obtener el estado del orquestador: {e}")
            return

        # 2. Insertar vectores a través del Orquestador
        log_info(f"\n--- Insertando {TOTAL_VECTORS_TO_INSERT} vectores a través del Orquestador ---")
        vectors_to_insert = generate_random_vectors(TOTAL_VECTORS_TO_INSERT, VECTOR_DIM)
        
        for i, vec in enumerate(vectors_to_insert):
            try:
                response = requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec.to_dict(), timeout=5)
                response.raise_for_status()
                # log_info(f"Vector {vec.id} insertado. Respuesta: {response.json()}")
            except requests.exceptions.RequestException as e:
                log_error(f"Error al insertar vector {vec.id} a través del Orquestador: {e}")
                log_error(f"Deteniendo la inserción. Verifica si los workers están levantados y registrados.")
                break
            if (i + 1) % (TOTAL_VECTORS_TO_INSERT // 5) == 0:
                log_info(f"  Insertados {i+1}/{TOTAL_VECTORS_TO_INSERT} vectores...")
        
        time.sleep(2) # Dar tiempo para que los workers procesen las inserciones

        # 3. Realizar búsqueda k-NN a través del Orquestador
        log_info(f"\n--- Realizando búsqueda k-NN (k={K_NN_SEARCH}) a través del Orquestador ---")
        query_vector = generate_random_vectors(1, VECTOR_DIM, start_id=TOTAL_VECTORS_TO_INSERT + 1)[0]
        log_info(f"Vector de consulta: {query_vector.data.tolist()[:5]}...")

        try:
            response = requests.post(
                f"{ORCHESTRATOR_URL}/search/knn",
                json={"query_vector": query_vector.to_dict(), "k": K_NN_SEARCH},
                timeout=10 # Aumentar timeout para búsquedas distribuidas
            )
            response.raise_for_status()
            search_results_raw = response.json().get("results", [])
            log_info(f"Resultados de búsqueda k-NN obtenidos ({len(search_results_raw)}):")
            for res_data in search_results_raw:
                res = SearchResult.from_dict(res_data)
                log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}") # Metadatos no se muestran para brevedad
            
            # Opcional: Verificar que los resultados se ordenan correctamente por distancia
            sorted_results = sorted(search_results_raw, key=lambda x: x['distance'])
            if len(search_results_raw) > 1 and sorted_results[0]['distance'] > sorted_results[-1]['distance']:
                 log_warning("Los resultados no están ordenados por distancia (Ascendente).")

        except requests.exceptions.RequestException as e:
            log_error(f"Error al realizar búsqueda k-NN a través del Orquestador: {e}")

        # 4. Realizar una búsqueda de rango
        log_info(f"\n--- Realizando búsqueda por rango (radio=1.5) a través del Orquestador ---")
        query_vector_range = generate_random_vectors(1, VECTOR_DIM, start_id=TOTAL_VECTORS_TO_INSERT + 2)[0]
        search_radius = 1.5

        try:
            response = requests.post(
                f"{ORCHESTRATOR_URL}/search/range",
                json={"query_vector": query_vector_range.to_dict(), "radius": search_radius},
                timeout=10
            )
            response.raise_for_status()
            search_results_range_raw = response.json().get("results", [])
            log_info(f"Resultados de búsqueda por rango obtenidos ({len(search_results_range_raw)}):")
            for res_data in search_results_range_raw:
                res = SearchResult.from_dict(res_data)
                log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al realizar búsqueda por rango a través del Orquestador: {e}")


    finally:
        if orchestrator_proc and worker_procs:
            stop_system(orchestrator_proc, worker_procs)
        log_info("--- Demostración de interacción Orquestador-Workers completada ---")

if __name__ == "__main__":
    run_orchestrator_interaction_demo()