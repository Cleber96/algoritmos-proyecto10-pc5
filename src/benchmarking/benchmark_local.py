# src/benchmarking/benchmark_local.py
import time
import os
import sys
import numpy as np
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt
import pandas as pd

# Añadir la ruta raíz del proyecto para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.m_tree.m_tree import MTree
from src.common.models import Vector, SearchResult
from src.common.utils import logger, euclidean_distance
from src.simulation.data_generator import generate_random_vectors

# Opcionales para comparación
try:
    from sklearn.neighbors import KDTree
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn (KDTree) no está instalado. No se realizará la comparación.")
    SKLEARN_AVAILABLE = False

# Aunque FAISS es un binario, su API Python es muy útil.
# Si lo instalas, puedes añadirlo para una comparación más robusta con un SOTA.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS no está instalado. No se realizará la comparación con FAISS.")
    FAISS_AVAILABLE = False


# --- Configuración del Benchmark ---
BENCHMARK_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'benchmark_results')
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

# Tamaños de dataset a probar
DATASET_SIZES = [1000, 5000, 10000, 20000] # Puedes aumentar para pruebas más extensas
VECTOR_DIM = 64
NUM_QUERIES = 50
K_NN_SEARCH = 10

def run_benchmark(dataset_size: int, vector_dim: int, num_queries: int, k_nn: int):
    """
    Ejecuta un benchmark comparando el M-Tree con otras implementaciones locales.
    """
    print(f"\n--- Ejecutando benchmark local para Dataset Size: {dataset_size} ---")
    
    # Generar datos
    data_vectors = generate_random_vectors(dataset_size, vector_dim)
    query_vectors = generate_random_vectors(num_queries, vector_dim, start_id=dataset_size + 1)

    results: Dict[str, Dict[str, float]] = {} # {implementation_name: {operation: time}}

    # --- M-Tree Benchmark ---
    print("Benchmarking M-Tree...")
    m_tree = MTree(max_children=4, min_children=2, distance_metric="euclidean")

    start_time = time.perf_counter()
    for vec in data_vectors:
        m_tree.insert(vec)
    insert_time_m_tree = time.perf_counter() - start_time
    print(f"  M-Tree Inserción ({dataset_size} vectores): {insert_time_m_tree:.4f} s")

    start_time = time.perf_counter()
    for query_vec in query_vectors:
        m_tree.search_knn(query_vec, k_nn)
    search_time_m_tree = (time.perf_counter() - start_time) / num_queries
    print(f"  M-Tree Búsqueda k-NN (avg {num_queries} queries): {search_time_m_tree:.6f} s/query")
    results['M-Tree'] = {'Insert': insert_time_m_tree, 'Search_kNN': search_time_m_tree}

    # --- KDTree (Scikit-learn) Benchmark ---
    if SKLEARN_AVAILABLE:
        print("Benchmarking KDTree (Scikit-learn)...")
        # KDTree no tiene una operación de "inserción" incremental como un árbol,
        # se construye de una vez.
        sklearn_data = np.array([vec.data for vec in data_vectors])

        start_time = time.perf_counter()
        kdtree = KDTree(sklearn_data, leaf_size=40) # leaf_size es un parámetro de optimización
        build_time_kdtree = time.perf_counter() - start_time
        print(f"  KDTree Construcción ({dataset_size} vectores): {build_time_kdtree:.4f} s")

        start_time = time.perf_counter()
        for query_vec in query_vectors:
            # query devuelve (distancias, indices)
            kdtree.query(query_vec.data.reshape(1, -1), k=k_nn)
        search_time_kdtree = (time.perf_counter() - start_time) / num_queries
        print(f"  KDTree Búsqueda k-NN (avg {num_queries} queries): {search_time_kdtree:.6f} s/query")
        results['KDTree'] = {'Build': build_time_kdtree, 'Search_kNN': search_time_kdtree}

    # --- FAISS (Flat index) Benchmark (Ejemplo básico) ---
    if FAISS_AVAILABLE:
        print("Benchmarking FAISS (Flat Index)...")
        # Faiss IndexFlatL2 es un índice que hace búsqueda de fuerza bruta (exacta)
        faiss_data = np.array([vec.data for vec in data_vectors]).astype('float32')
        
        start_time = time.perf_counter()
        # d es la dimensión de los vectores
        index = faiss.IndexFlatL2(vector_dim) 
        index.add(faiss_data)
        build_time_faiss = time.perf_counter() - start_time
        print(f"  FAISS Construcción ({dataset_size} vectores): {build_time_faiss:.4f} s")

        start_time = time.perf_counter()
        for query_vec in query_vectors:
            D, I = index.search(query_vec.data.reshape(1, -1).astype('float32'), k_nn)
        search_time_faiss = (time.perf_counter() - start_time) / num_queries
        print(f"  FAISS Búsqueda k-NN (avg {num_queries} queries): {search_time_faiss:.6f} s/query")
        results['FAISS_Flat'] = {'Build': build_time_faiss, 'Search_kNN': search_time_faiss}

    return results

def plot_results(all_results: Dict[int, Dict[str, Dict[str, float]]], operation: str):
    """
    Genera gráficos a partir de los resultados del benchmark.
    """
    plt.figure(figsize=(10, 6))
    
    for impl_name in all_results[DATASET_SIZES[0]].keys(): # Asumiendo que todas las implementaciones tienen los mismos resultados
        x = []
        y = []
        for size in DATASET_SIZES:
            if impl_name in all_results[size] and operation in all_results[size][impl_name]:
                x.append(size)
                y.append(all_results[size][impl_name][operation])
        plt.plot(x, y, label=impl_name, marker='o')

    plt.title(f'Rendimiento de {operation} vs. Tamaño del Dataset')
    plt.xlabel('Número de Vectores en el Dataset')
    plt.ylabel(f'Tiempo (segundos)')
    plt.xticks(DATASET_SIZES)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log') # Puedes usar escala logarítmica para datasets grandes
    plt.yscale('log') # Opcional: para ver patrones en un rango amplio de tiempos
    
    plot_filename = os.path.join(BENCHMARK_RESULTS_DIR, f'benchmark_local_{operation.lower()}_plot.png')
    plt.savefig(plot_filename)
    print(f"Gráfico guardado en: {plot_filename}")
    plt.close()

if __name__ == '__main__':
    all_benchmark_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for size in DATASET_SIZES:
        all_benchmark_results[size] = run_benchmark(size, VECTOR_DIM, NUM_QUERIES, K_NN_SEARCH)
    
    print("\n--- Resumen de Resultados del Benchmark Local ---")
    df_insert = pd.DataFrame({size: {impl: res.get('Insert', np.nan) for impl, res in data.items()} 
                              for size, data in all_benchmark_results.items()})
    df_search = pd.DataFrame({size: {impl: res.get('Search_kNN', np.nan) for impl, res in data.items()} 
                              for size, data in all_benchmark_results.items()})
    
    print("\nTiempos de Inserción/Construcción:")
    print(df_insert.to_markdown(floatfmt=".4f"))
    
    print("\nTiempos de Búsqueda k-NN (por query):")
    print(df_search.to_markdown(floatfmt=".6f"))

    # Generar gráficos
    plot_results(all_benchmark_results, 'Insert')
    plot_results(all_benchmark_results, 'Search_kNN')

    print("\nBenchmark local completado. Consulta los archivos .prof y .png para análisis detallado.")