# src/common/utils.py
import numpy as np
import logging
from typing import Callable # Importar Callable para tipado de funciones
from .models import Vector # Importar la clase Vector si se utiliza en alguna función de utilidad

# Configuración básica de logging
# Asegurarse de que el logger se configure una sola vez al inicio de la aplicación
# Si se llama varias veces, puede duplicar los handlers.
# Una mejor práctica para módulos es solo obtener el logger y configurarlo en un script principal.
# Para propósitos de desarrollo/testing aquí, está bien.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def euclidean_distance(vec1_data: np.ndarray, vec2_data: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre dos arrays de numpy.
    """
    if vec1_data.shape != vec2_data.shape:
        raise ValueError("Los vectores deben tener la misma dimensión para calcular la distancia euclidiana.")
    return np.linalg.norm(vec1_data - vec2_data).item() # .item() para obtener un float escalar nativo

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

    if norm_a == 0.0 or norm_b == 0.0: 
        # Si uno o ambos vectores son el vector cero, la similitud es indefinida.
        # Convencionalmente se retorna 0.0, o se podría levantar un error.
        logger.warning("Uno o ambos vectores son cero en similitud coseno. Retornando 0.0.")
        return 0.0 
    
    return dot_product / (norm_a * norm_b)

def get_distance_metric(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Retorna la función de métrica de distancia apropiada basada en el nombre.
    Si se solicita 'cosine', retorna 1 - similitud coseno para usarla como una "distancia".
    """
    if metric_name.lower() == 'euclidean':
        return euclidean_distance
    elif metric_name.lower() == 'cosine':
        # Para que el M-Tree funcione correctamente con la similitud coseno,
        # necesitamos transformarla en una métrica de distancia donde un valor
        # más pequeño signifique "más cercano" (mayor similitud).
        # Usamos 1 - similitud_coseno.
        # Similitud 1.0 (idéntico) -> Distancia 0.0
        # Similitud 0.0 (ortogonal) -> Distancia 1.0
        # Similitud -1.0 (opuesto) -> Distancia 2.0
        return lambda v1, v2: 1 - cosine_similarity(v1, v2)
    else:
        raise ValueError(f"Métrica de distancia '{metric_name}' no soportada.")

# Funciones de logging con los nombres de tu propuesta original para consistencia
def log_info(message: str):
    logger.info(message)

def log_warning(message: str):
    logger.warning(message)

def log_error(message: str):
    logger.error(message)