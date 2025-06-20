# src/profiling/profile_mt_insert.py
import cProfile
import pstats
import os
import sys
import time
import numpy as np

# Añadir la ruta raíz del proyecto para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.m_tree.m_tree import MTree
from src.common.models import Vector
from src.common.utils import logger
from src.simulation.data_generator import generate_random_vectors

# --- Configuración del Perfilado ---
PROFILE_DIR = os.path.join(os.path.dirname(__file__), 'profile_results')
os.makedirs(PROFILE_DIR, exist_ok=True)

NUM_VECTORS_FOR_PROFILE = 5000 # Número de vectores para la inserción masiva
VECTOR_DIM = 64              # Dimensión de los vectores

# Desactivar logging excesivo durante el perfilado para no distorsionar resultados
logger.disabled = True

def run_profiled_m_tree_insertion():
    """
    Perfilado de la operación de inserción masiva en el M-Tree.
    """
    print(f"\n--- Perfilando inserción masiva en M-Tree de {NUM_VECTORS_FOR_PROFILE} vectores ---")
    
    vectors_to_insert = generate_random_vectors(NUM_VECTORS_FOR_PROFILE, VECTOR_DIM)
    
    # Instancia del M-Tree
    m_tree = MTree(max_children=4, min_children=2, distance_metric="euclidean")

    profile_path = os.path.join(PROFILE_DIR, 'mtree_insert_profile.prof')
    profiler = cProfile.Profile()
    profiler.enable()

    for i, vec in enumerate(vectors_to_insert):
        m_tree.insert(vec)
        if (i + 1) % 1000 == 0:
            print(f"  Insertados {i+1} vectores...")

    profiler.disable()
    profiler.dump_stats(profile_path)
    print(f"Resultados de perfilado de inserción M-Tree guardados en: {profile_path}")

    # Mostrar las 10 funciones más costosas en la consola
    stats = pstats.Stats(profile_path)
    stats.sort_stats('cumulative').print_stats(10) # 'cumulative' o 'tottime'
    
    logger.disabled = False # Reactivar logging
    logger.info("Perfilado de inserción M-Tree completado.")


if __name__ == "__main__":
    run_profiled_m_tree_insertion()
    print("\nPara visualizar los resultados detalladamente, usa snakeviz:")
    print(f"  snakeviz {os.path.join(PROFILE_DIR, 'mtree_insert_profile.prof')}")
    print("\nAsegúrate de tener snakeviz instalado: pip install snakeviz")