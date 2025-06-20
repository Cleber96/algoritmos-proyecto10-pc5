# src/profiling/profile_worker.py
import cProfile
import pstats
import os
import sys
import time
import json
import requests
import numpy as np

# Añadir la ruta raíz del proyecto para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.worker_node.app import app as worker_app # Importar la app de Flask del worker
from src.worker_node.worker_service import WorkerService # Para instanciar directamente
from src.common.models import Vector
from src.common.utils import logger, get_distance_metric
from src.simulation.data_generator import generate_random_vectors # Necesitamos un generador de datos

# --- Configuración del Perfilado ---
PROFILE_DIR = os.path.join(os.path.dirname(__file__), 'profile_results')
os.makedirs(PROFILE_DIR, exist_ok=True)

NUM_VECTORS_INSERT = 1000 # Número de vectores para insertar y perfilar
NUM_QUERIES_KNN = 100      # Número de búsquedas k-NN a perfilar
K_NN = 10                  # Valor de K para k-NN
VECTOR_DIM = 64            # Dimensión de los vectores

# Desactivar logging excesivo durante el perfilado para no distorsionar resultados
logger.disabled = True

def run_profiled_worker_operations():
    """
    Ejecuta operaciones clave del WorkerService bajo cProfile.
    No ejecuta la aplicación Flask, sino directamente la lógica del servicio.
    """
    logger.info("Iniciando perfilado de WorkerService...")
    
    # Simular la configuración del M-Tree
    m_tree_config = {
        "max_children": 4,
        "min_children": 2,
        "distance_metric": "euclidean"
    }
    
    # Instanciar el WorkerService directamente
    worker = WorkerService(node_id="profile_worker", m_tree_config=m_tree_config)

    # 1. Perfilado de Inserción Masiva
    print(f"\n--- Perfilando Inserción de {NUM_VECTORS_INSERT} vectores ---")
    vectors_to_insert = generate_random_vectors(NUM_VECTORS_INSERT, VECTOR_DIM)
    
    insert_profile_path = os.path.join(PROFILE_DIR, 'worker_insert_profile.prof')
    profiler = cProfile.Profile()
    profiler.enable()

    for vec in vectors_to_insert:
        worker.insert_vector(vec.to_dict())

    profiler.disable()
    profiler.dump_stats(insert_profile_path)
    print(f"Resultados de perfilado de inserción guardados en: {insert_profile_path}")

    # 2. Perfilado de Búsqueda k-NN Masiva
    print(f"\n--- Perfilando Búsqueda k-NN de {NUM_QUERIES_KNN} queries ---")
    query_vectors = generate_random_vectors(NUM_QUERIES_KNN, VECTOR_DIM, start_id=NUM_VECTORS_INSERT + 1)
    
    search_profile_path = os.path.join(PROFILE_DIR, 'worker_search_knn_profile.prof')
    profiler = cProfile.Profile()
    profiler.enable()

    for query_vec in query_vectors:
        worker.search_knn(query_vec.to_dict(), K_NN)

    profiler.disable()
    profiler.dump_stats(search_profile_path)
    print(f"Resultados de perfilado de búsqueda k-NN guardados en: {search_profile_path}")

    # 3. Perfilado de Búsqueda por Rango Masiva (Opcional, similar a k-NN)
    # Puedes añadir un perfilado similar para search_range si es relevante.

    logger.info("Perfilado de WorkerService completado.")
    logger.disabled = False # Reactivar logging


if __name__ == "__main__":
    run_profiled_worker_operations()

    print("\nPara visualizar los resultados, usa snakeviz:")
    print(f"  snakeviz {os.path.join(PROFILE_DIR, 'worker_insert_profile.prof')}")
    print(f"  snakeviz {os.path.join(PROFILE_DIR, 'worker_search_knn_profile.prof')}")
    print("\nAsegúrate de tener snakeviz instalado: pip install snakeviz")