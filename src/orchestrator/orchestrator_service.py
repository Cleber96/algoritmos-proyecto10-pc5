# src/orchestrator/orchestrator_service.py
import requests
import json
from typing import List, Dict, Any, Optional
from src.consistent_hashing.consistent_hasher import ConsistentHasher
from src.common.models import Vector, SearchResult
from src.common.utils import logger
import threading
import time

class OrchestratorService:
    """
    Servicio orquestador que maneja la distribución de vectores
    y las peticiones de búsqueda a los nodos worker.
    """
    def __init__(self, hasher_replicas: int = 100):
        self.consistent_hasher = ConsistentHasher(replicas=hasher_replicas)
        self.worker_nodes: Dict[str, str] = {} # {node_id: node_url}
        logger.info(f"OrchestratorService inicializado.")
        # Opcional: un bloqueo para asegurar consistencia al añadir/eliminar nodos
        self.node_lock = threading.Lock() 

    def register_worker(self, node_id: str, node_url: str) -> bool:
        """
        Registra un nodo worker con el orquestador.
        Añade el nodo al anillo de hashing consistente.
        """
        with self.node_lock:
            if node_id in self.worker_nodes:
                logger.warning(f"Worker '{node_id}' ya registrado con URL {self.worker_nodes[node_id]}. Actualizando URL a {node_url}.")
                # Posibilidad de actualizar URL si el worker cambia de ubicación
                self.worker_nodes[node_id] = node_url
            else:
                self.worker_nodes[node_id] = node_url
                self.consistent_hasher.add_node(node_id)
                logger.info(f"Worker '{node_id}' registrado con URL {node_url}.")
            return True

    def deregister_worker(self, node_id: str) -> bool:
        """
        Elimina un nodo worker del orquestador y del anillo de hashing.
        """
        with self.node_lock:
            if node_id not in self.worker_nodes:
                logger.warning(f"Worker '{node_id}' no encontrado para desregistro.")
                return False
            
            del self.worker_nodes[node_id]
            self.consistent_hasher.remove_node(node_id)
            logger.info(f"Worker '{node_id}' desregistrado.")
            return True

    def insert_vector(self, vector_data: Dict[str, Any]) -> bool:
        """
        Recibe un vector, determina el worker responsable y lo envía para inserción.
        """
        vector_id = vector_data.get("id")
        if not vector_id:
            logger.error("Vector data missing 'id' for insertion.")
            return False

        target_node_id = self.consistent_hasher.get_node(vector_id)
        if not target_node_id:
            logger.error(f"No worker available to handle vector '{vector_id}'.")
            return False
        
        target_node_url = self.worker_nodes.get(target_node_id)
        if not target_node_url:
            logger.error(f"URL for worker '{target_node_id}' not found. Cannot insert vector '{vector_id}'.")
            # Esto puede ocurrir si el nodo fue desregistrado justo después de get_node
            self.consistent_hasher.remove_node(target_node_id) # Eliminar nodo roto
            return False

        try:
            # Asegurarse de enviar el vector en el formato esperado por el worker
            headers = {'Content-Type': 'application/json'}
            response = requests.post(f"{target_node_url}/insert", json=vector_data, headers=headers, timeout=5)
            response.raise_for_status() # Lanza excepción para códigos de estado 4xx/5xx
            logger.info(f"Vector '{vector_id}' enviado a worker '{target_node_id}'. Response: {response.json().get('status')}")
            return response.json().get("status") == "success"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error inserting vector '{vector_id}' to worker '{target_node_id}' at {target_node_url}: {e}")
            # Considerar desregistrar el nodo si la comunicación falla persistentemente
            # self.deregister_worker(target_node_id)
            return False

    def search_knn(self, query_vector_data: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda k-NN distribuyendo la petición a todos los workers
        y agregando los resultados.
        """
        query_id = query_vector_data.get("id", "N/A_Query")
        logger.info(f"Orchestrator: Initiating distributed k-NN search for query '{query_id}' (k={k}).")

        all_results: List[SearchResult] = []
        worker_futures = []
        
        # Para evitar problemas con el cambio de nodos durante la iteración
        active_workers = list(self.worker_nodes.items()) 

        # Podrías usar ThreadPoolExecutor para paralelizar las peticiones a workers
        # For simplicity, we'll do it sequentially or use basic threading if needed.
        for node_id, node_url in active_workers:
            try:
                headers = {'Content-Type': 'application/json'}
                payload = {"query_vector": query_vector_data, "k": k}
                response = requests.post(f"{node_url}/search/knn", json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                
                worker_k_results_dicts = response.json().get("results", [])
                
                # Convertir dicts a SearchResult objects
                worker_k_results = [SearchResult.from_dict(res_dict) for res_dict in worker_k_results_dicts]
                
                all_results.extend(worker_k_results)
                logger.debug(f"Worker '{node_id}' returned {len(worker_k_results)} results for '{query_id}'.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error searching k-NN on worker '{node_id}' at {node_url}: {e}")
                # Considerar desregistrar el nodo
                # self.deregister_worker(node_id)

        # Agregación: Obtener los k mejores resultados de todos los workers
        # Necesitamos ordenar por distancia y tomar los k primeros.
        all_results.sort(key=lambda x: x.distance)
        final_k_results = all_results[:k]

        logger.info(f"Orchestrator: Distributed k-NN search for '{query_id}' completed. Found {len(final_k_results)} global results.")
        return [res.to_dict() for res in final_k_results]

    def search_range(self, query_vector_data: Dict[str, Any], radius: float) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda por rango, distribuyendo la petición a todos los workers
        y agregando los resultados (sin duplicados, si es posible).
        """
        query_id = query_vector_data.get("id", "N/A_Query")
        logger.info(f"Orchestrator: Initiating distributed range search for query '{query_id}' (radius={radius}).")

        all_results: List[SearchResult] = []
        # Usamos un set para evitar IDs duplicados si la misma query llega a varios workers
        # y un vector se inserta cerca del "borde" del hashing consistente, duplicándolo.
        # En el M-Tree puro esto no pasaría, pero en un sistema distribuido puede haber replicación
        # o rangos superpuestos por si acaso.
        unique_result_ids: Set[str] = set()

        active_workers = list(self.worker_nodes.items())

        for node_id, node_url in active_workers:
            try:
                headers = {'Content-Type': 'application/json'}
                payload = {"query_vector": query_vector_data, "radius": radius}
                response = requests.post(f"{node_url}/search/range", json=payload, headers=headers, timeout=10)
                response.raise_for_status()

                worker_range_results_dicts = response.json().get("results", [])
                
                for res_dict in worker_range_results_dicts:
                    if res_dict["vector_id"] not in unique_result_ids:
                        all_results.append(SearchResult.from_dict(res_dict))
                        unique_result_ids.add(res_dict["vector_id"])
                
                logger.debug(f"Worker '{node_id}' returned {len(worker_range_results_dicts)} results for '{query_id}'.")

            except requests.exceptions.RequestException as e:
                logger.error(f"Error searching range on worker '{node_id}' at {node_url}: {e}")
                # Considerar desregistrar el nodo

        all_results.sort(key=lambda x: x.distance)
        logger.info(f"Orchestrator: Distributed range search for '{query_id}' completed. Found {len(all_results)} global unique results.")
        return [res.to_dict() for res in all_results]

    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado general del sistema, incluyendo el estado de cada worker
        y el anillo de hashing.
        """
        status = {
            "orchestrator_status": "READY",
            "active_workers_count": len(self.worker_nodes),
            "consistent_hasher_status": self.consistent_hasher.get_ring_status(),
            "worker_details": []
        }
        
        # Obtener el estado de cada worker
        worker_statuses = []
        for node_id, node_url in list(self.worker_nodes.items()): # Copia para evitar problemas si un nodo se desregistra
            try:
                response = requests.get(f"{node_url}/status", timeout=2)
                response.raise_for_status()
                worker_statuses.append(response.json())
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get status from worker '{node_id}' at {node_url}: {e}")
                worker_statuses.append({"node_id": node_id, "status": "UNREACHABLE", "error": str(e)})
                # Opcional: Deregistrar nodos no alcanzables aquí, o tener un mecanismo de monitoreo separado
                # self.deregister_worker(node_id)
        
        status["worker_details"] = worker_statuses
        return status