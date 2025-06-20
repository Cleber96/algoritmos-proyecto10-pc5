# src/worker_node/app.py
from flask import Flask, request, jsonify
import os
import sys

# Asegúrate de que `src` esté en el PATH para las importaciones relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.worker_node.worker_service import WorkerService
from src.common.utils import logger
from src.common.models import Vector

app = Flask(__name__)

# Configuración del M-Tree (podría venir de config/settings.py)
M_TREE_CONFIG = {
    "max_children": 4, # Ajusta según el rendimiento deseado
    "min_children": 2,
    "distance_metric": "euclidean"
}

# Inicializa el servicio del worker
worker_service: Optional[WorkerService] = None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de salud del worker."""
    return jsonify({"status": "healthy", "node_id": worker_service.node_id if worker_service else "N/A"}), 200

@app.route('/status', methods=['GET'])
def get_worker_status():
    """Endpoint para obtener el estado detallado del worker y su M-Tree."""
    if not worker_service:
        return jsonify({"error": "Worker service not initialized"}), 500
    return jsonify(worker_service.get_status()), 200

@app.route('/insert', methods=['POST'])
def insert_vector():
    """Endpoint para insertar un vector."""
    if not worker_service:
        return jsonify({"error": "Worker service not initialized"}), 500

    data = request.get_json()
    if not data or 'id' not in data or 'vector' not in data:
        return jsonify({"error": "Missing 'id' or 'vector' in request body"}), 400
    
    try:
        if worker_service.insert_vector(data):
            return jsonify({"status": "success", "message": f"Vector {data['id']} inserted"}), 201
        else:
            return jsonify({"status": "error", "message": f"Failed to insert vector {data['id']}"}), 500
    except Exception as e:
        logger.error(f"API Error inserting vector: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search/knn', methods=['POST'])
def search_knn():
    """Endpoint para buscar k-NN."""
    if not worker_service:
        return jsonify({"error": "Worker service not initialized"}), 500

    data = request.get_json()
    if not data or 'query_vector' not in data or 'k' not in data:
        return jsonify({"error": "Missing 'query_vector' or 'k' in request body"}), 400
    
    k = data['k']
    query_vector_data = data['query_vector']

    try:
        results = worker_service.search_knn(query_vector_data, k)
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        logger.error(f"API Error during k-NN search: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search/range', methods=['POST'])
def search_range():
    """Endpoint para buscar por rango."""
    if not worker_service:
        return jsonify({"error": "Worker service not initialized"}), 500

    data = request.get_json()
    if not data or 'query_vector' not in data or 'radius' not in data:
        return jsonify({"error": "Missing 'query_vector' or 'radius' in request body"}), 400
    
    radius = data['radius']
    query_vector_data = data['query_vector']

    try:
        results = worker_service.search_range(query_vector_data, radius)
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        logger.error(f"API Error during range search: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # El ID del worker y el puerto se pasan como argumentos de línea de comandos
    # Esto es crucial para la simulación distribuida.
    import argparse
    parser = argparse.ArgumentParser(description="Worker Node for Distributed Vector Search")
    parser.add_argument("--id", type=str, required=True, help="Unique ID for this worker node (e.g., worker_1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on")
    args = parser.parse_args()

    NODE_ID = args.id
    PORT = args.port

    worker_service = WorkerService(NODE_ID, M_TREE_CONFIG)
    logger.info(f"Starting Worker Node '{NODE_ID}' on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False) # debug=True solo para desarrollo local