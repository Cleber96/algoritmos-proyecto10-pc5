# src/orchestrator/app.py
from flask import Flask, request, jsonify
import os
import sys

# Asegúrate de que `src` esté en el PATH para las importaciones relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.orchestrator.orchestrator_service import OrchestratorService
from src.common.utils import logger
from src.common.models import Vector # Para validar datos de entrada

app = Flask(__name__)

# Inicializa el servicio del orquestador
ORCHESTRATOR_HASHER_REPLICAS = 100 # Número de réplicas virtuales para Consistent Hashing
orchestrator_service = OrchestratorService(hasher_replicas=ORCHESTRATOR_HASHER_REPLICAS)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de salud del orquestador."""
    return jsonify({"status": "healthy", "service": "orchestrator"}), 200

@app.route('/status', methods=['GET'])
def get_system_status():
    """Endpoint para obtener el estado completo del sistema distribuido."""
    return jsonify(orchestrator_service.get_system_status()), 200

@app.route('/register_worker', methods=['POST'])
def register_worker():
    """Endpoint para que los workers se registren con el orquestador."""
    data = request.get_json()
    if not data or 'node_id' not in data or 'node_url' not in data:
        return jsonify({"error": "Missing 'node_id' or 'node_url' in request body"}), 400
    
    node_id = data['node_id']
    node_url = data['node_url']
    
    if orchestrator_service.register_worker(node_id, node_url):
        return jsonify({"status": "success", "message": f"Worker '{node_id}' registered"}), 200
    else:
        return jsonify({"status": "error", "message": f"Failed to register worker '{node_id}'"}), 500

@app.route('/deregister_worker', methods=['POST'])
def deregister_worker():
    """Endpoint para desregistrar un worker."""
    data = request.get_json()
    if not data or 'node_id' not in data:
        return jsonify({"error": "Missing 'node_id' in request body"}), 400
    
    node_id = data['node_id']
    
    if orchestrator_service.deregister_worker(node_id):
        return jsonify({"status": "success", "message": f"Worker '{node_id}' deregistered"}), 200
    else:
        return jsonify({"status": "error", "message": f"Failed to deregister worker '{node_id}'"}), 500

@app.route('/insert_vector', methods=['POST'])
def insert_vector():
    """Endpoint para insertar un vector en el sistema distribuido."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must contain vector data"}), 400
    
    # Valida el formato del vector usando la clase Vector
    try:
        Vector.from_dict(data) 
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid vector data: {e}"}), 400

    if orchestrator_service.insert_vector(data):
        return jsonify({"status": "success", "message": f"Vector {data.get('id', 'N/A')} sent for insertion"}), 202 # Aceptado para procesamiento
    else:
        return jsonify({"status": "error", "message": f"Failed to distribute or insert vector {data.get('id', 'N/A')}"}), 500

@app.route('/search/knn', methods=['POST'])
def search_knn():
    """Endpoint para realizar una búsqueda k-NN en el sistema distribuido."""
    data = request.get_json()
    if not data or 'query_vector' not in data or 'k' not in data:
        return jsonify({"error": "Missing 'query_vector' or 'k' in request body"}), 400
    
    k = data['k']
    query_vector_data = data['query_vector']

    # Valida el formato del query_vector
    try:
        Vector.from_dict(query_vector_data)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid query_vector data: {e}"}), 400

    try:
        results = orchestrator_service.search_knn(query_vector_data, k)
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        logger.error(f"Orchestrator API Error during distributed k-NN search: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search/range', methods=['POST'])
def search_range():
    """Endpoint para realizar una búsqueda por rango en el sistema distribuido."""
    data = request.get_json()
    if not data or 'query_vector' not in data or 'radius' not in data:
        return jsonify({"error": "Missing 'query_vector' or 'radius' in request body"}), 400
    
    radius = data['radius']
    query_vector_data = data['query_vector']

    # Valida el formato del query_vector
    try:
        Vector.from_dict(query_vector_data)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid query_vector data: {e}"}), 400

    try:
        results = orchestrator_service.search_range(query_vector_data, radius)
        return jsonify({"status": "success", "results": results}), 200
    except Exception as e:
        logger.error(f"Orchestrator API Error during distributed range search: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # El puerto del orquestador se define aquí o se pasa como argumento
    ORCHESTRATOR_PORT = 5000 # O 8000, 8080, etc.

    logger.info(f"Starting Orchestrator Node on port {ORCHESTRATOR_PORT}...")
    app.run(host='0.0.0.0', port=ORCHESTRATOR_PORT, debug=False) # debug=True solo para desarrollo local