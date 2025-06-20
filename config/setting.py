# config/settings.py
import os

# Configuración del M-Tree
M_TREE_CONFIG = {
    "max_children": int(os.environ.get("M_TREE_MAX_CHILDREN", 4)),
    "min_children": int(os.environ.get("M_TREE_MIN_CHILDREN", 2)),
    "distance_metric": os.environ.get("M_TREE_DISTANCE_METRIC", "euclidean").lower()
}

# Configuración del Hashing Consistente
CONSISTENT_HASHER_REPLICAS = int(os.environ.get("CONSISTENT_HASHER_REPLICAS", 100))

# Configuración de red del Orquestador
ORCHESTRATOR_HOST = os.environ.get("ORCHESTRATOR_HOST", "0.0.0.0")
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", 5000)) # Puerto por defecto del orquestador

# Configuración de red de los Workers (los puertos se asignarán dinámicamente o por script)
WORKER_BASE_PORT = int(os.environ.get("WORKER_BASE_PORT", 5001)) # Puerto inicial para workers

# Configuración general
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()