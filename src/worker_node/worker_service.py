# src/worker_node/worker_service.py
from typing import List, Dict, Any, Optional
from src.m_tree.m_tree import MTree
from src.common.models import Vector, SearchResult
from src.common.utils import logger, get_distance_metric
import json

class WorkerService:
    """
    Servicio que gestiona las operaciones de M-Tree para un nodo worker.
    """
    def __init__(self, node_id: str, m_tree_config: Dict[str, Any]):
        self.node_id = node_id
        # Inicializa el M-Tree con la configuración proporcionada
        self.m_tree = MTree(
            max_children=m_tree_config.get("max_children", 4),
            min_children=m_tree_config.get("min_children", 2),
            distance_metric=m_tree_config.get("distance_metric", "euclidean")
        )
        logger.info(f"WorkerService '{self.node_id}' inicializado con M-Tree.")
        self.status = "READY"

    def insert_vector(self, vector_data: Dict[str, Any]) -> bool:
        """
        Inserta un vector en el M-Tree de este worker.
        """
        try:
            vector = Vector.from_dict(vector_data)
            self.m_tree.insert(vector)
            logger.info(f"Worker '{self.node_id}': Vector '{vector.id}' insertado. M-Tree size: {self.m_tree.get_size()}")
            return True
        except Exception as e:
            logger.error(f"Worker '{self.node_id}': Error al insertar vector '{vector_data.get('id', 'N/A')}': {e}")
            return False

    def search_knn(self, query_vector_data: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda k-NN en el M-Tree de este worker.
        """
        try:
            query_vector = Vector.from_dict(query_vector_data)
            results: List[SearchResult] = self.m_tree.search_knn(query_vector, k)
            logger.info(f"Worker '{self.node_id}': Búsqueda k-NN para '{query_vector.id}' (k={k}) completada. Encontrados {len(results)} resultados.")
            return [res.to_dict() for res in results]
        except Exception as e:
            logger.error(f"Worker '{self.node_id}': Error al realizar búsqueda k-NN para '{query_vector_data.get('id', 'N/A')}': {e}")
            return []

    def search_range(self, query_vector_data: Dict[str, Any], radius: float) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por rango en el M-Tree de este worker.
        """
        try:
            query_vector = Vector.from_dict(query_vector_data)
            results: List[SearchResult] = self.m_tree.search_range(query_vector, radius)
            logger.info(f"Worker '{self.node_id}': Búsqueda por rango para '{query_vector.id}' (radius={radius}) completada. Encontrados {len(results)} resultados.")
            return [res.to_dict() for res in results]
        except Exception as e:
            logger.error(f"Worker '{self.node_id}': Error al realizar búsqueda por rango para '{query_vector_data.get('id', 'N/A')}': {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del worker y de su M-Tree."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "m_tree_size": self.m_tree.get_size(),
            "m_tree_metrics": self.m_tree.metrics_counter
        }