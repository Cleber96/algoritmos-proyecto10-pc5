# src/simulation/data_generator.py
import numpy as np
import json
import os
import uuid # Para generar IDs únicos
from typing import List, Dict, Any

# Ajusta el nivel de importación para que funcione cuando se ejecuta directamente y como módulo
try:
    from src.common.models import Vector
except ImportError:
    # Fallback para cuando se ejecuta directamente desde src/simulation
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.common.models import Vector

def generate_random_vectors(
    num_vectors: int,
    vector_dim: int,
    output_path: str,
    vector_type: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    mean: float = 0.0,
    std_dev: float = 1.0
) -> None:
    """
    Genera un conjunto de vectores aleatorios y los guarda en un archivo JSON.
    Los vectores son instancias de la clase Vector definida en common/models.py.

    Args:
        num_vectors (int): Número de vectores a generar.
        vector_dim (int): Dimensión de cada vector.
        output_path (str): Ruta donde se guardará el archivo JSON.
        vector_type (str): Tipo de distribución ('uniform' o 'gaussian').
        low (float): Límite inferior para la distribución uniforme.
        high (float): Límite superior para la distribución uniforme.
        mean (float): Media para la distribución gaussiana.
        std_dev (float): Desviación estándar para la distribución gaussiana.
    """
    print(f"Generando {num_vectors} vectores de {vector_dim} dimensiones (tipo: {vector_type})...")
    vectors_data: List[Dict[str, Any]] = []

    for i in range(num_vectors):
        if vector_type == "uniform":
            data = np.random.uniform(low, high, vector_dim).tolist()
        elif vector_type == "gaussian":
            data = np.random.normal(mean, std_dev, vector_dim).tolist()
        else:
            raise ValueError("Tipo de vector no válido. Use 'uniform' o 'gaussian'.")

        # Crea una instancia de Vector
        vec_obj = Vector(
            id=str(uuid.uuid4()), # Genera un ID único para cada vector
            data=data,
            metadata={
                "index": i,
                "type": vector_type,
                "source": "generated_data"
            }
        )
        vectors_data.append(vec_obj.to_dict())

    # Asegurarse de que el directorio de salida exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(vectors_data, f, indent=2)

    print(f"Vectores generados y guardados en: {output_path}")

if __name__ == "__main__":
    # --- Configuración de ejemplo ---
    # Para la práctica, es recomendable generar un número significativo de vectores
    # para probar la distribución y el rendimiento del M-Tree y el hashing consistente.
    NUM_VECTORS = 100000       # Un buen número para empezar a ver el rendimiento distribuido
    VECTOR_DIMENSION = 128     # Dimensión común para embeddings (ej. de BERT, imagen)
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../../data/sample_vectors.json')

    # Generar vectores uniformes (excelente para pruebas iniciales de distribución)
    print("\n--- Generando Vectores Uniformes ---")
    generate_random_vectors(NUM_VECTORS, VECTOR_DIMENSION, OUTPUT_FILE, vector_type="uniform", low=-1.0, high=1.0)

    # Opcional: Generar vectores gaussianos (podrían simular clusters naturales, interesantes para M-Tree)
    # Si quieres generar un segundo set, cambia el nombre del archivo de salida.
    # GAUSSIAN_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../../data/gaussian_vectors.json')
    # print("\n--- Generando Vectores Gaussianos ---")
    # generate_random_vectors(NUM_VECTORS, VECTOR_DIMENSION, GAUSSIAN_OUTPUT_FILE, vector_type="gaussian", mean=0.0, std_dev=0.5)   