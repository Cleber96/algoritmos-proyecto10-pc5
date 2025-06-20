# demos/demo_m_tree_local.py
import sys
import os
import numpy as np
import random

# Añadir la ruta raíz del proyecto al sys.path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.m_tree.m_tree import MTree
from src.common.models import Vector, SearchResult
from src.common.utils import get_distance_metric, log_info, log_warning

def run_m_tree_demo():
    """
    Demostración de la funcionalidad básica del M-Tree:
    inserción de vectores y búsqueda k-NN.
    """
    log_info("--- Iniciando demostración local del M-Tree ---")

    # Configuración del M-Tree
    # Puedes probar con "euclidean" o "cosine" aquí
    DISTANCE_METRIC = "euclidean" 
    try:
        distance_fn = get_distance_metric(DISTANCE_METRIC)
    except ValueError as e:
        log_error(f"Error al obtener la métrica de distancia: {e}")
        return

    m_tree = MTree(max_children=4, min_children=2, distance_metric=DISTANCE_METRIC)
    log_info(f"M-Tree inicializado con max_children=4, min_children=2 y métrica '{DISTANCE_METRIC}'.")

    # 1. Inserción de vectores de ejemplo
    log_info("\n--- Insertando vectores de ejemplo en el M-Tree ---")
    vectors_to_insert = [
        Vector("v1", [1.0, 1.0, 1.0]),
        Vector("v2", [2.0, 2.0, 2.0]),
        Vector("v3", [1.1, 1.2, 1.0]),
        Vector("v4", [5.0, 5.0, 5.0]),
        Vector("v5", [5.1, 5.2, 5.0]),
        Vector("v6", [10.0, 10.0, 10.0]),
        Vector("v7", [10.1, 9.9, 10.2]),
        Vector("v8", [0.5, 0.5, 0.5]),
        Vector("v9", [7.0, 7.0, 7.0]),
        Vector("v10", [7.1, 6.9, 7.2]),
    ]

    for vec in vectors_to_insert:
        m_tree.insert(vec)
        log_info(f"Vector insertado: {vec.id}")
    
    log_info(f"Total de vectores en el M-Tree: {m_tree.size()}")
    if m_tree.root:
        log_info(f"Profundidad del M-Tree: {m_tree.root.get_height()}")
    else:
        log_warning("El árbol está vacío o la raíz no está inicializada.")


    # 2. Búsqueda k-NN
    log_info("\n--- Realizando búsquedas k-NN ---")

    # Consulta 1: Cerca de v1, v2, v3
    query_vector1 = Vector("q1", [1.05, 1.1, 1.0])
    k_nn1 = 3
    log_info(f"Buscando los {k_nn1} vecinos más cercanos a {query_vector1.data.tolist()} (Query: {query_vector1.id})")
    results1 = m_tree.search_knn(query_vector1, k_nn1)
    for res in results1:
        log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Datos: {res.vector.data.tolist()}")

    # Consulta 2: Cerca de v6, v7
    query_vector2 = Vector("q2", [10.05, 10.0, 10.1])
    k_nn2 = 2
    log_info(f"\nBuscando los {k_nn2} vecinos más cercanos a {query_vector2.data.tolist()} (Query: {query_vector2.id})")
    results2 = m_tree.search_knn(query_vector2, k_nn2)
    for res in results2:
        log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Datos: {res.vector.data.tolist()}")

    # Consulta 3: Un vector que no está en el árbol
    query_vector3 = Vector("q3", [0.0, 0.0, 0.0])
    k_nn3 = 1
    log_info(f"\nBuscando el {k_nn3} vecino más cercano a {query_vector3.data.tolist()} (Query: {query_vector3.id})")
    results3 = m_tree.search_knn(query_vector3, k_nn3)
    if results3:
        for res in results3:
            log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Datos: {res.vector.data.tolist()}")
    else:
        log_info("  No se encontraron vecinos (esto podría indicar un árbol vacío o error en la búsqueda si los datos están muy dispersos).")
    
    # 3. Búsqueda por Rango
    log_info("\n--- Realizando búsquedas por Rango ---")
    query_vector_range = Vector("qr1", [1.0, 1.0, 1.0])
    search_radius = 0.5
    log_info(f"Buscando vectores dentro de un radio de {search_radius} alrededor de {query_vector_range.data.tolist()}")
    results_range = m_tree.search_range(query_vector_range, search_radius)
    if results_range:
        for res in results_range:
            log_info(f"  - Vecino: {res.vector.id}, Distancia: {res.distance:.4f}, Datos: {res.vector.data.tolist()}")
    else:
        log_info("  No se encontraron vectores dentro del rango especificado.")


    log_info("\n--- Demostración del M-Tree completada ---")

if __name__ == "__main__":
    run_m_tree_demo()