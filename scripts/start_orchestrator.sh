#!/bin/bash
# scripts/start_orchestrator.sh

echo "Iniciando el nodo Orquestador..."

# Activa el entorno virtual
source .venv/bin/activate

# Ejecuta la aplicación Flask/FastAPI del orquestador
# Asegúrate de que el puerto aquí coincide con ORCHESTRATOR_PORT en config/settings.py o en app.py
python src/orchestrator/app.py --port 5000 # O el puerto que hayas definido

# Desactiva el entorno virtual al salir (opcional, si quieres que el script termine su ejecución limpia)
# deactivate