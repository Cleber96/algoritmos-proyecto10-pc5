# src/benchmarking/benchmark_distributed.py
import os
import sys
import time
import subprocess
import requests
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd

# Añadir la ruta raíz del proyecto para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.models import Vector, SearchResult
from src.common.utils import logger
from src.simulation.data_generator import generate_random_vectors

# --- Configuración del Benchmark Distribuido ---
BENCHMARK_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results')
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

ORCHESTRATOR_PORT = 5000
WORKER_BASE_PORT = 5001
ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"

# Parámetros de prueba
NUM_WORKERS_CONFIGS = [1, 2, 4] # Cantidad de workers a probar
TOTAL_VECTORS_INSERT = 10000    # Total de vectores a insertar en el sistema
NUM_DISTRIBUTED_QUERIES = 200   # Número de queries a realizar
K_NN_SEARCH = 10
VECTOR_DIM = 64

# Retraso para asegurar que los servicios estén listos
STARTUP_DELAY = 5 # segundos

def start_system(num_workers: int, orchestrator_port: int, worker_base_port: int) -> Tuple[subprocess.Popen, List[subprocess.Popen]]:
    """Inicia el orquestador y los workers."""
    logger.info(f"Iniciando sistema con {num_workers} workers...")
    
    orchestrator_proc = subprocess.Popen(
        ['python', 'src/orchestrator/app.py', '--port', str(orchestrator_port)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    logger.info(f"Orquestador iniciado (PID: {orchestrator_proc.pid}).")
    time.sleep(STARTUP_DELAY) # Dale tiempo para iniciar

    worker_procs: List[subprocess.Popen] = []
    worker_urls: List[str] = []

    for i in range(num_workers):
        worker_id = f"worker_{i+1}"
        worker_port = worker_base_port + i
        worker_url = f"http://localhost:{worker_port}"
        
        proc = subprocess.Popen(
            ['python', 'src/worker_node/app.py', '--id', worker_id, '--port', str(worker_port)],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        worker_procs.append(proc)
        worker_urls.append(worker_url)
        logger.info(f"Worker '{worker_id}' iniciado (PID: {proc.pid}) en {worker_url}.")

    time.sleep(STARTUP_DELAY) # Dale tiempo a los workers para iniciar

    # Registrar workers con el orquestador
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
            logger.info(f"Worker '{worker_id}' registrado con Orquestador: {response.json().get('status')}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al registrar worker '{worker_id}': {e}")

    time.sleep(1) # Pequeña pausa después del registro
    return orchestrator_proc, worker_procs

def stop_system(orchestrator_proc: subprocess.Popen, worker_procs: List[subprocess.Popen]):
    """Detiene todos los procesos del sistema."""
    logger.info("Deteniendo el sistema distribuido...")
    
    # Terminar workers primero
    for proc in worker_procs:
        if proc.poll() is None: # Si el proceso sigue corriendo
            proc.terminate()
            proc.wait(timeout=2)
            if proc.poll() is None: # Si no terminó, mátalo
                proc.kill()
        logger.debug(f"Worker PID {proc.pid} detenido.")

    # Terminar orquestador
    if orchestrator_proc.poll() is None:
        orchestrator_proc.terminate()
        orchestrator_proc.wait(timeout=2)
        if orchestrator_proc.poll() is None:
            orchestrator_proc.kill()
    logger.debug(f"Orquestador PID {orchestrator_proc.pid} detenido.")
    
    logger.info("Sistema detenido.")

def run_distributed_benchmark():
    """
    Ejecuta el benchmark distribuido para diferentes configuraciones de workers.
    """
    all_results: Dict[int, Dict[str, float]] = {} # {num_workers: {operation_type: time}}

    for num_workers in NUM_WORKERS_CONFIGS:
        orchestrator_proc, worker_procs = None, []
        try:
            # 1. Iniciar el sistema
            orchestrator_proc, worker_procs = start_system(num_workers, ORCHESTRATOR_PORT, WORKER_BASE_PORT)
            
            # 2. Verificar el estado del sistema (opcional pero bueno para depuración)
            try:
                status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=5)
                status_response.raise_for_status()
                logger.info(f"Estado del sistema con {num_workers} workers: {status_response.json().get('active_workers_count')} workers activos.")
            except Exception as e:
                logger.error(f"Error al obtener el estado del orquestador: {e}")
                # Si el orquestador no responde, es un problema grave, aborta.
                raise

            # 3. Benchmark de Inserción Masiva
            print(f"\n--- Benchmark de Inserción para {num_workers} Workers ---")
            vectors_to_insert = generate_random_vectors(TOTAL_VECTORS_INSERT, VECTOR_DIM)
            
            insert_start_time = time.perf_counter()
            for i, vec in enumerate(vectors_to_insert):
                try:
                    response = requests.post(f"{ORCHESTRATOR_URL}/insert_vector", json=vec.to_dict(), timeout=5)
                    response.raise_for_status()
                    if response.json().get("status") != "success" and response.json().get("status") != "accepted":
                        logger.warning(f"Fallo al insertar vector {vec.id}: {response.json().get('message')}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de red al insertar vector {vec.id}: {e}")
                if (i + 1) % (TOTAL_VECTORS_INSERT // 10) == 0:
                    print(f"  Insertados {i+1}/{TOTAL_VECTORS_INSERT} vectores...")
            insert_end_time = time.perf_counter()
            total_insert_time = insert_end_time - insert_start_time
            avg_insert_latency = total_insert_time / TOTAL_VECTORS_INSERT
            print(f"  Tiempo total de inserción ({TOTAL_VECTORS_INSERT} vectores): {total_insert_time:.4f} s")
            print(f"  Latencia promedio de inserción: {avg_insert_latency:.6f} s/vector")
            
            # Asegurar que todos los vectores fueron procesados (dar tiempo a los workers)
            time.sleep(2) 
            
            # Opcional: Verificar tamaños en workers
            # try:
            #     status_response = requests.get(f"{ORCHESTRATOR_URL}/status", timeout=5)
            #     total_size = sum(w['m_tree_size'] for w in status_response.json()['worker_details'] if w['status'] == 'READY')
            #     print(f"  Total de vectores en M-Trees de Workers: {total_size}")
            # except Exception: pass


            # 4. Benchmark de Búsqueda k-NN Masiva
            print(f"\n--- Benchmark de Búsqueda k-NN para {num_workers} Workers ---")
            query_vectors = generate_random_vectors(NUM_DISTRIBUTED_QUERIES, VECTOR_DIM, start_id=TOTAL_VECTORS_INSERT + 1)
            
            search_start_time = time.perf_counter()
            for i, query_vec in enumerate(query_vectors):
                try:
                    response = requests.post(f"{ORCHESTRATOR_URL}/search/knn", json={"query_vector": query_vec.to_dict(), "k": K_NN_SEARCH}, timeout=10)
                    response.raise_for_status()
                    # Opcional: Validar resultados si es necesario
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de red al buscar k-NN para query {query_vec.id}: {e}")
                if (i + 1) % (NUM_DISTRIBUTED_QUERIES // 5) == 0:
                    print(f"  Realizadas {i+1}/{NUM_DISTRIBUTED_QUERIES} búsquedas...")
            search_end_time = time.perf_counter()
            total_search_time = search_end_time - search_start_time
            avg_search_latency = total_search_time / NUM_DISTRIBUTED_QUERIES
            print(f"  Tiempo total de búsqueda k-NN ({NUM_DISTRIBUTED_QUERIES} queries): {total_search_time:.4f} s")
            print(f"  Latencia promedio de búsqueda k-NN: {avg_search_latency:.6f} s/query")

            all_results[num_workers] = {
                'Insert_Total': total_insert_time,
                'Insert_Avg_Latency': avg_insert_latency,
                'Search_Total': total_search_time,
                'Search_Avg_Latency': avg_search_latency
            }

        except Exception as e:
            logger.critical(f"Fallo crítico durante el benchmark distribuido para {num_workers} workers: {e}", exc_info=True)
            all_results[num_workers] = {
                'Insert_Total': np.nan, 'Insert_Avg_Latency': np.nan,
                'Search_Total': np.nan, 'Search_Avg_Latency': np.nan
            }
        finally:
            # 5. Detener el sistema
            if orchestrator_proc and worker_procs:
                stop_system(orchestrator_proc, worker_procs)
            time.sleep(2) # Pequeña pausa entre configuraciones para asegurar limpieza

    return all_results

def plot_distributed_results(results: Dict[int, Dict[str, float]], metric_key: str, title_suffix: str, ylabel: str):
    """Genera gráficos para los resultados distribuidos."""
    plt.figure(figsize=(8, 5))
    
    num_workers_list = sorted(list(results.keys()))
    metric_values = [results[nw].get(metric_key, np.nan) for nw in num_workers_list]

    plt.plot(num_workers_list, metric_values, marker='o', linestyle='-')
    plt.title(f'Rendimiento Distribuido: {title_suffix}')
    plt.xlabel('Número de Workers')
    plt.ylabel(ylabel)
    plt.xticks(num_workers_list)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_filename = os.path.join(BENCHMARK_RESULTS_DIR, f'benchmark_distributed_{metric_key.lower()}_plot.png')
    plt.savefig(plot_filename)
    print(f"Gráfico guardado en: {plot_filename}")
    plt.close()


if __name__ == '__main__':
    logger.info("Iniciando benchmark del sistema distribuido...")
    
    # Asegúrate de que las apps de Flask no usen debug=True en producción/benchmarking
    # Esto puede hacerse configurando la variable de entorno FLASK_ENV=production
    # o directamente en los archivos app.py.

    benchmark_results = run_distributed_benchmark()
    
    print("\n--- Resumen de Resultados del Benchmark Distribuido ---")
    df_distributed = pd.DataFrame(benchmark_results).T # Transponer para tener workers como índice
    print(df_distributed.to_markdown(floatfmt=".6f"))

    # Generar gráficos
    plot_distributed_results(benchmark_results, 'Insert_Total', 'Tiempo Total de Inserción', 'Tiempo Total (s)')
    plot_distributed_results(benchmark_results, 'Insert_Avg_Latency', 'Latencia Promedio de Inserción', 'Latencia por Vector (s)')
    plot_distributed_results(benchmark_results, 'Search_Total', 'Tiempo Total de Búsqueda k-NN', 'Tiempo Total (s)')
    plot_distributed_results(benchmark_results, 'Search_Avg_Latency', 'Latencia Promedio de Búsqueda k-NN', 'Latencia por Query (s)')

    logger.info("Benchmark distribuido completado. Consulta los archivos en 'benchmark_results' para análisis.")