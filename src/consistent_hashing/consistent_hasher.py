# src/consistent_hashing/consistent_hasher.py
import hashlib
from typing import List, Dict, Union, Callable
from bisect import bisect_left, bisect_right
from src.common.utils import logger
from src.common.models import Vector

class ConsistentHasher:
    """
    Implementa un anillo de Hashing Consistente para distribuir claves
    (IDs de vectores) entre nodos de servicio (workers).
    """
    def __init__(self, replicas: int = 100):
        self.replicas = replicas  # Número de réplicas virtuales por nodo físico
        self.ring: List[int] = []  # Anillo de hash ordenado (puntos de hash)
        self.node_map: Dict[int, str] = {}  # Mapeo de puntos de hash a nombres de nodo
        self.nodes: Dict[str, bool] = {} # Nodos activos: {node_name: True}
        logger.info(f"ConsistentHasher inicializado con {replicas} réplicas virtuales por nodo.")

    def _hash(self, key: str) -> int:
        """Función de hash SHA-1 para convertir una cadena en un entero."""
        return int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % (2**32) # Usamos 2**32 para un rango de enteros grande pero manejable

    def add_node(self, node_name: str):
        """
        Añade un nuevo nodo al anillo de hashing consistente.
        Genera `self.replicas` puntos virtuales para el nodo.
        """
        if node_name in self.nodes:
            logger.warning(f"Nodo '{node_name}' ya existe en el hasher.")
            return

        self.nodes[node_name] = True
        for i in range(self.replicas):
            virtual_node_key = f"{node_name}-{i}"
            hash_point = self._hash(virtual_node_key)
            
            # Inserta el punto de hash en el anillo de manera ordenada
            # y mapea este punto al nombre del nodo
            # Usar bisect_left para mantener el orden
            idx = bisect_left(self.ring, hash_point)
            self.ring.insert(idx, hash_point)
            self.node_map[hash_point] = node_name
        
        logger.info(f"Nodo '{node_name}' añadido con {self.replicas} réplicas.")

    def remove_node(self, node_name: str):
        """
        Elimina un nodo del anillo de hashing consistente.
        Remueve todos sus puntos virtuales.
        """
        if node_name not in self.nodes:
            logger.warning(f"Nodo '{node_name}' no encontrado en el hasher para eliminación.")
            return

        del self.nodes[node_name]
        points_to_remove = []
        for hash_point, mapped_node_name in self.node_map.items():
            if mapped_node_name == node_name:
                points_to_remove.append(hash_point)
        
        for hash_point in points_to_remove:
            self.ring.remove(hash_point)
            del self.node_map[hash_point]
        
        logger.info(f"Nodo '{node_name}' eliminado. Removidos {len(points_to_remove)} réplicas.")

    def get_node(self, key: str) -> Optional[str]:
        """
        Obtiene el nodo responsable para una clave dada.
        Retorna el nombre del nodo.
        """
        if not self.ring:
            logger.warning("Anillo de hash vacío. No hay nodos para asignar la clave.")
            return None

        key_hash = self._hash(key)
        
        # Encuentra el primer punto de hash en el anillo que es mayor o igual al hash de la clave.
        # Si no se encuentra, se "envuelve" al principio del anillo.
        idx = bisect_left(self.ring, key_hash)
        
        if idx == len(self.ring):
            idx = 0 # Envolver al principio del anillo
        
        assigned_hash_point = self.ring[idx]
        assigned_node = self.node_map[assigned_hash_point]
        
        logger.debug(f"Key '{key}' (hash {key_hash}) asignada a node '{assigned_node}' (hash point {assigned_hash_point}).")
        return assigned_node

    def get_all_nodes(self) -> List[str]:
        """Retorna una lista de todos los nodos físicos actualmente en el anillo."""
        return list(self.nodes.keys())

    def get_ring_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del anillo para depuración/monitoreo."""
        status = {
            "num_physical_nodes": len(self.nodes),
            "num_virtual_points": len(self.ring),
            "physical_nodes": list(self.nodes.keys()),
            # Para evitar un output muy grande, no listamos todos los puntos
            # "ring_points": sorted(list(self.node_map.keys())) # Si necesitas ver los puntos
        }
        # Podrías añadir distribución de claves si quieres analizar el balanceo
        return status