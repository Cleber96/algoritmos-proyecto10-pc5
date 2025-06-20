# src/common/models.py
import numpy as np
from typing import List, Dict, Any, Union
import uuid

class Vector:
    """
    Representa un vector de alta dimensión con un ID único y metadatos opcionales.
    """
    def __init__(self, id: str, data: Union[List[float], np.ndarray], metadata: Dict[str, Any] = None):
        if not isinstance(id, str) or not id:
            raise ValueError("Vector ID must be a non-empty string.")
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                self.data = data.astype(np.float32)
            else:
                self.data = data
        else:
            raise TypeError("Vector data must be a list of floats or a numpy array.")

        if not self.data.ndim == 1:
            raise ValueError("Vector data must be a 1-dimensional array.")

        self.id = id
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el objeto Vector a un diccionario para serialización."""
        return {
            "id": self.id,
            "vector": self.data.tolist(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vector':
        """Crea un objeto Vector desde un diccionario."""
        if "id" not in data or "vector" not in data:
            raise ValueError("Dictionary must contain 'id' and 'vector' keys.")
        return cls(data["id"], data["vector"], data.get("metadata"))

    @property
    def dim(self) -> int:
        """Retorna la dimensión del vector."""
        return len(self.data)

    def __repr__(self) -> str:
        return f"Vector(id='{self.id}', dim={self.dim}, metadata={self.metadata})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.id == other.id and np.array_equal(self.data, other.data)

    def __hash__(self) -> int:
        return hash(self.id)


class SearchResult:
    """
    Representa un resultado de búsqueda, incluyendo el vector encontrado y su distancia.
    """
    def __init__(self, vector: Vector, distance: float):
        if not isinstance(vector, Vector):
            raise TypeError("SearchResult 'vector' must be an instance of Vector.")
        if not isinstance(distance, (int, float)):
            raise TypeError("SearchResult 'distance' must be a number.")

        self.vector = vector
        self.distance = float(distance) # Asegura que sea float

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el objeto SearchResult a un diccionario para serialización."""
        return {
            "vector_id": self.vector.id,
            "distance": self.distance,
            # Incluir un snippet del vector y sus metadatos para contexto
            "vector_data_snippet": self.vector.data[:5].tolist() + ['...'] if len(self.vector.data) > 5 else self.vector.data.tolist(),
            "metadata": self.vector.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Crea un objeto SearchResult desde un diccionario."""
        if "vector_id" not in data or "distance" not in data:
            raise ValueError("Dictionary must contain 'vector_id' and 'distance' keys.")
        # Nota: Para reconstruir el Vector completo, necesitarías el 'vector_data' completo
        # Aquí solo reconstruimos un Vector parcial para el SearchResult si el `vector_data` no está completo.
        # En un sistema real, solo pasarías el ID y la distancia, y el cliente consultaría el vector completo si lo necesita.
        # Para la simulación, podemos asumir que el snippet es suficiente para la representación.
        vector_data = data.get("vector_data_snippet", []) # Asumiendo que es un snippet si no está completo
        if isinstance(vector_data, list) and '...' in vector_data:
            # Si es un snippet, no podemos reconstruir el vector completo, creamos uno dummy o lo omitimos.
            # Para este caso, simplificamos creando un Vector con data vacía, ya que solo el ID y dist importan para el resultado.
            dummy_vector = Vector(id=data["vector_id"], data=[0.0] * 1) # Vector con datos dummy
            dummy_vector.metadata = data.get("metadata", {})
            return cls(dummy_vector, data["distance"])
        else:
            # Si el snippet es el vector completo (ej. para vectores de baja dim), o si pasas el 'vector' completo
            full_vector_data = data.get("vector_data", vector_data) # Intenta obtener 'vector_data' completo
            full_vector = Vector(id=data["vector_id"], data=full_vector_data, metadata=data.get("metadata", {}))
            return cls(full_vector, data["distance"])


    def __repr__(self) -> str:
        return f"SearchResult(vector_id='{self.vector.id}', distance={self.distance:.4f})"