#!/bin/bash
# scripts/run_all.sh

NUM_WORKERS=${1:-3}
ORCHESTRATOR_PORT=${2:-5000}
WORKER_BASE_PORT=${3:-5001}

echo "Iniciando el sistema distribuido de búsqueda de vectores..."

# Activa el entorno virtual
source .venv/bin/activate

# Detener cualquier proceso python existente para asegurar un inicio limpio
echo "Deteniendo procesos Python anteriores..."
killall python 2>/dev/null || true # Ignora errores si no hay procesos

sleep 1

# 1. Iniciar el Orquestador en segundo plano
echo "Iniciando Orquestador en puerto $ORCHESTRATOR_PORT..."
python src/orchestrator/app.py --port "$ORCHESTRATOR_PORT" &
ORCHESTRATOR_PID=$!
echo "Orquestador iniciado (PID: $ORCHESTRATOR_PID)."

# Espera un momento para que el orquestador inicie completamente
sleep 5

# 2. Iniciar y registrar Workers
echo "Iniciando y registrando $NUM_WORKERS Workers..."
./scripts/start_workers.sh $NUM_WORKERS $WORKER_BASE_PORT

echo "Todos los componentes del sistema deberían estar en funcionamiento."
echo "Puedes verificar el estado del sistema con curl http://localhost:$ORCHESTRATOR_PORT/status"
echo "Para detener todos los procesos, usa: kill $ORCHESTRATOR_PID; killall python"

# Mantener el script en ejecución para mantener los procesos activos.
# Opcional: esperar a que el usuario presione una tecla para salir
# read -p "Presiona Enter para detener el sistema..."
# kill $ORCHESTRATOR_PID
# killall python
# deactivate