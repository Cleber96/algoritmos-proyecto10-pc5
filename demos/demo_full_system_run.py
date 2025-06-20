# demos/demo_full_system_run.py
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
from src.simulation.data_generator import generate_random_vectors, generate_cluster_vectors

ORCHESTRATOR_PORT = 5000
WORKER_BASE_PORT = 5001
ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"

# --- Configuración de la Simulación Completa ---
NUM_WORKERS_FOR_SIMULATION = 4
TOTAL_VECTORS_TO_INSERT_SIM = 500 # Un número razonable para una demo rápida
NUM_QUERY_VECTORS_SIM = 50
VECTOR_DIM_SIM = 64
K_NN_SIM = 5

STARTUP_DELAY_SYSTEM = 5 # segundos

def start_system(num_workers: int) -> tuple[subprocess.Popen, List[subprocess.Popen]]:
    """Inicia el orquestador y los workers."""
    log_info(f"Iniciando sistema completo con {num_workers} workers...")
    
    orchestrator_proc = subprocess.Popen(
        ['python', 'src/orchestrator/app.py', '--port', str(ORCHESTRATOR_PORT)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE # Capturar salida para evitar polución del stdout
    )
    log_info(f"Orquestador iniciado (PID: {orchestrator_proc.pid}).")
    time.sleep(STARTUP_DELAY_SYSTEM)

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

    time.sleep(STARTUP_DELAY_SYSTEM)

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
            log_error(f"Error crítico al registrar worker '{worker_id}': {e}")
            raise # Detener la demo si el registro falla

    time.sleep(1)
    return orchestrator_proc, worker_procs

def stop_system(orchestrator_proc: subprocess.Popen, worker_procs: List[subprocess.Popen]):
    """Detiene todos los procesos del sistema."""
    log_info("\n--- Deteniendo el sistema distribuido ---")
    
    for proc in worker_procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                log_warning(f"Worker PID {proc.pid} fue forzado a terminar.")
        log_info(f"Worker PID {proc.pid} detenido.")

    if orchestrator_proc.poll() is None:
        orchestrator_proc.terminate()
        try:
            orchestrator_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            orchestrator_proc.kill()
            log_warning(f"Orquestador PID {orchestrator_proc.pid} fue forzado a terminar.")
    log_info(f"Orquestador PID {orchestrator_proc.pid} detenido.")
    
    log_info("Sistema detenido completamente.")

def run_full_system_demo():
    """
    Ejecuta una demostración completa del sistema distribuido:
    inicia, inserta, busca y detiene.
    """
    log_info("--- Iniciando Demostración Completa del Sistema Distribuido ---")

    orchestrator_proc, worker_procs = None, []
    try:
        orchestrator_proc, worker_procs = start_system(NUM_WORKERS_FOR_SIMULATION)
        
        # 1. Verificar estado inicial
        log_info("\n--- Verificando estado inicial del Orquestador ---")
        try:
            status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=5)
            status_response.raise_for_status()
            log_info(f"Estado del Orquestador: {status_response.json()}")
        except requests.exceptions.RequestException as e:
            log_error(f"Error al obtener estado inicial del Orquestador: {e}")
            return

        # 2. Insertar una gran cantidad de vectores
        log_info(f"\n--- Insertando {TOTAL_VECTORS_TO_INSERT_SIM} vectores en el sistema distribuido ---")
        # Puedes usar generate_random_vectors o generate_cluster_vectors
        vectors_to_insert = generate_random_vectors(TOTAL_VECTORS_TO_INSERT_SIM, VECTOR_DIM_SIM)
        
        insert_start_time = time.perf_counter()
        for i, vec in enumerate(vectors_to_insert):
            try:
                response = requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec.to_dict(), timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                log_error(f"Error al insertar vector {vec.id}: {e}")
                break
            if (i + 1) % (TOTAL_VECTORS_TO_INSERT_SIM // 10) == 0:
                log_info(f"  Insertados {i+1}/{TOTAL_VECTORS_TO_INSERT_SIM} vectores...")
        insert_end_time = time.perf_counter()
        log_info(f"Tiempo total de inserción: {insert_end_time - insert_start_time:.4f} segundos")
        
        time.sleep(3) # Dar tiempo para que las inserciones se asienten en los workers

        # 3. Realizar múltiples búsquedas k-NN
        log_info(f"\n--- Realizando {NUM_QUERY_VECTORS_SIM} búsquedas k-NN (k={K_NN_SIM}) ---")
        query_vectors = generate_random_vectors(NUM_QUERY_VECTORS_SIM, VECTOR_DIM_SIM, start_id=TOTAL_VECTORS_TO_INSERT_SIM + 1)
        
        search_start_time = time.perf_counter()
        successful_queries = 0
        for i, query_vec in enumerate(query_vectors):
            try:
                response = requests.post(
                    f"{ORCHESTRATOR_URL}/search/knn",
                    json={"query_vector": query_vec.to_dict(), "k": K_NN_SIM},
                    timeout=15 # Aumentar timeout para búsquedas en grandes datasets distribuidos
                )
                response.raise_for_status()
                # log_info(f"  Query {query_vec.id}: {len(response.json().get('results', []))} resultados.")
                successful_queries += 1
            except requests.exceptions.RequestException as e:
                log_error(f"Error al buscar k-NN para query {query_vec.id}: {e}")
            if (i + 1) % (NUM_QUERY_VECTORS_SIM // 5) == 0:
                log_info(f"  Realizadas {i+1}/{NUM_QUERY_VECTORS_SIM} búsquedas...")
        search_end_time = time.perf_counter()
        log_info(f"Tiempo total de búsqueda ({successful_queries} exitosas): {search_end_time - search_start_time:.4f} segundos")
        
        # Opcional: Mostrar algunos resultados de la última búsqueda
        if successful_queries > 0 and 'response' in locals(): # Check if 'response' variable exists from last successful query
            last_results = response.json().get("results", [])
            if last_results:
                log_info(f"\nPrimeros 3 resultados de la última búsqueda:")
                for res_data in last_results[:3]:
                    res = SearchResult.from_dict(res_data)
                    log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Metadatos: {res.vector.metadata}")
            else:
                log_info("La última búsqueda no arrojó resultados.")

        # 4. Verificar el estado final del sistema (tamaños de M-Tree en cada worker)
        log_info("\n--- Verificando estado final de los Workers ---")
        try:
            status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=5)
            status_response.raise_for_status()
            worker_details = status_response.json().get('worker_details', [])
            total_vectors_in_workers = 0
            for worker in worker_details:
                log_info(f"  Worker {worker['node_id']} ({worker['node_url']}): {worker['m_tree_size']} vectores.")
                total_vectors_in_workers += worker['m_tree_size']
            log_info(f"Total de vectores reportados en Workers: {total_vectors_in_workers}")
            if total_vectors_in_workers != TOTAL_VECTORS_TO_INSERT_SIM:
                log_warning(f"Discrepancia en el total de vectores. Insertados: {TOTAL_VECTORS_TO_INSERT_SIM}, Reportados: {total_vectors_in_workers}")

        except requests.exceptions.RequestException as e:
            log_error(f"Error al obtener estado final del Orquestador/Workers: {e}")


    finally:
        if orchestrator_proc and worker_procs:
            stop_system(orchestrator_proc, worker_procs)
        log_info("--- Demostración Completa del Sistema Distribuido finalizada ---")

if __name__ == "__main__":
    run_full_system_demo()