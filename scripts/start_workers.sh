#!/bin/bash
# scripts/start_workers.sh

NUM_WORKERS=${1:-3} # Número de workers, por defecto 3
BASE_PORT=${2:-5001} # Puerto inicial para los workers

echo "Iniciando $NUM_WORKERS nodos Worker a partir del puerto $BASE_PORT..."

# Activa el entorno virtual
source .venv/bin/activate

ORCHESTRATOR_URL="http://localhost:5000" # URL del orquestador

for i in $(seq 1 $NUM_WORKERS); do
    WORKER_ID="worker_$i"
    WORKER_PORT=$((BASE_PORT + i - 1))
    WORKER_URL="http://localhost:$WORKER_PORT"

    echo "Iniciando Worker '$WORKER_ID' en el puerto $WORKER_PORT..."
    
    # Inicia el worker en segundo plano
    python src/worker_node/app.py --id "$WORKER_ID" --port "$WORKER_PORT" &
    WORKER_PID=$! # Guarda el PID del worker
    echo "Worker '$WORKER_ID' (PID: $WORKER_PID) iniciado."

    # Dale un momento al worker para que inicie y esté listo
    sleep 2 

    # Registra el worker con el orquestador
    echo "Registrando Worker '$WORKER_ID' con el Orquestador en $ORCHESTRATOR_URL..."
    # Usamos curl para enviar la petición de registro
    curl -X POST -H "Content-Type: application/json" \
         -d "{\"node_id\": \"$WORKER_ID\", \"node_url\": \"$WORKER_URL\"}" \
         "$ORCHESTRATOR_URL/register_worker"
    echo "" # Salto de línea después de la salida de curl
done

echo "Todos los Workers iniciados y registrados (si el orquestador estaba activo)."
echo "Para detener los workers, usa 'killall python' o busca los PIDs."

# Para mantener el entorno virtual activo si vas a interactuar con los workers manualmente después
# deactivate