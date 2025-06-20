# src/common/models.py
import numpy as np
from typing import List, Dict, Any, Union
import uuid # Aunque no se usa directamente en Vector, es buena práctica tenerla si la tenías por algo más.

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
        # Comparamos IDs y los datos del vector.
        return self.id == other.id and np.array_equal(self.data, other.data)

    def __hash__(self) -> int:
        # Usamos solo el ID para el hash, asumiendo que es único.
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
        """
        Crea un objeto SearchResult desde un diccionario.
        Intenta reconstruir el Vector completo si se proporcionan los datos.
        """
        if "vector_id" not in data or "distance" not in data:
            raise ValueError("Dictionary must contain 'vector_id' and 'distance' keys.")
        
        # Preferimos 'vector' (datos completos) si está disponible, sino 'vector_data_snippet'.
        # Esto es importante para reconstruir el objeto Vector correctamente.
        vector_data = data.get("vector") 
        if vector_data is None:
            vector_data = data.get("vector_data_snippet", []) 
            if '...' in vector_data:
                 # Si solo hay un snippet y no los datos completos, creamos un vector con datos dummy.
                 # En un escenario real, si se necesita el vector completo, se haría otra llamada.
                 logger.warning(f"Reconstruyendo SearchResult para {data['vector_id']} con datos de vector parciales/dummy.")
                 dummy_vector = Vector(id=data["vector_id"], data=np.array([0.0], dtype=np.float32)) 
                 dummy_vector.metadata = data.get("metadata", {})
                 return cls(dummy_vector, data["distance"])
        
        # Si tenemos los datos completos o un snippet que es el vector completo.
        full_vector = Vector(id=data["vector_id"], data=vector_data, metadata=data.get("metadata", {}))
        return cls(full_vector, data["distance"])

    def __repr__(self) -> str:
        return f"SearchResult(vector_id='{self.vector.id}', distance={self.distance:.4f})"

    def __lt__(self, other: Any) -> bool:
        """Permite comparar SearchResults por distancia (útil para heaps/ordenamiento)."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.distance < other.distance