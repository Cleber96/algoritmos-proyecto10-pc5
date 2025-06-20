# src/common/utils.py
import numpy as np
import logging
from typing import List
from .models import Vector # Importar la clase Vector

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def euclidean_distance(vec1_data: np.ndarray, vec2_data: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre dos arrays de numpy.
    """
    if vec1_data.shape != vec2_data.shape:
        raise ValueError("Los vectores deben tener la misma dimensión para calcular la distancia euclidiana.")
    return np.linalg.norm(vec1_data - vec2_data).item() # .item() para obtener un float escalar

def cosine_similarity(vec1_data: np.ndarray, vec2_data: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos arrays de numpy.
    Retorna un valor entre -1 y 1. Mayor valor = más similar.
    """
    if vec1_data.shape != vec2_data.shape:
        raise ValueError("Los vectores deben tener la misma dimensión para calcular la similitud coseno.")
    dot_product = np.dot(vec1_data, vec2_data)
    norm_a = np.linalg.norm(vec1_data)
    norm_b = np.linalg.norm(vec2_data)

    if norm_a == 0 or norm_b == 0:
        return 0.0 # O NaN, dependiendo de cómo quieras manejar vectores cero
    return dot_product / (norm_a * norm_b)

def get_distance_metric(metric_name: str):
    """
    Retorna la función de métrica de distancia basada en el nombre.
    """
    if metric_name.lower() == 'euclidean':
        return euclidean_distance
    elif metric_name.lower() == 'cosine':
        # Para búsqueda de similitud, a veces se usa 1 - coseno para convertirlo en "distancia"
        # Aquí retornamos la similitud coseno directamente, la inversión se haría en la lógica de búsqueda.
        return cosine_similarity
    else:
        raise ValueError(f"Métrica de distancia '{metric_name}' no soportada.")

# Ejemplo de uso de logging
def log_info(message: str):
    logger.info(message)

def log_warning(message: str):
    logger.warning(message)

def log_error(message: str):
    logger.error(message)