#!/bin/bash

# Nombre del entorno virtual
ENV_NAME=".venv"

echo "==================================================="
echo "Configurando el entorno de desarrollo para Distributed Vector Search"
echo "==================================================="

# 1. Verificar si Python está instalado
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 no está instalado. Por favor, instálalo para continuar."
    exit 1
fi

# 2. Crear el entorno virtual si no existe
if [ ! -d "$ENV_NAME" ]; then
    echo "Creando entorno virtual '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error al crear el entorno virtual. Asegúrate de tener python3-venv instalado (sudo apt-get install python3-venv en Debian/Ubuntu)."
        exit 1
    fi
    echo "Entorno virtual creado exitosamente."
else
    echo "El entorno virtual '$ENV_NAME' ya existe."
fi

# 3. Activar el entorno virtual
echo "Activando el entorno virtual..."
source "$ENV_NAME"/bin/activate
if [ $? -ne 0 ]; then
    echo "Error al activar el entorno virtual."
    exit 1
fi
echo "Entorno virtual activado."

# 4. Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Advertencia: No se pudo actualizar pip. Continuando con la instalación de dependencias."
    # No salimos aquí, ya que a veces es un problema de permisos menores o red y el resto puede funcionar
fi


# 5. Instalar dependencias desde requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error al instalar dependencias desde requirements.txt. Por favor, revisa el archivo."
        exit 1
    fi
    echo "Dependencias de requirements.txt instaladas."
else
    echo "Advertencia: El archivo requirements.txt no se encontró. No se instalarán dependencias básicas."
fi

# 6. (Opcional) Instalar dependencias de desarrollo usando Poetry o pip de pyproject.toml
# Si estás usando pyproject.toml con herramientas como Poetry o directamente para flit/hatch
# y quieres instalar dependencias de desarrollo, puedes agregar lógica aquí.
# Para este proyecto, asumiendo que pyproject.toml se usa para black/ruff/pydantic
# y que estas pueden listarse también en requirements.txt o instalarse directamente.
# Aquí un ejemplo para pyproject.toml con ruff y black si no están en requirements.txt:
# if [ -f "pyproject.toml" ]; then
#    echo "Instalando herramientas de desarrollo (ruff, black) si no están ya en requirements.txt..."
#    pip install ruff black pydantic
#    if [ $? -ne 0 ]; then
#        echo "Advertencia: No se pudieron instalar algunas herramientas de desarrollo."
#    fi
# fi

echo "==================================================="
echo "Configuración del entorno completada."
echo "Para trabajar en el entorno virtual, usa:"
echo "source $ENV_NAME/bin/activate"
echo "Cuando hayas terminado, puedes usar 'deactivate'"
echo "==================================================="

# No desactivar el entorno aquí, para que el usuario pueda seguir trabajando en él
# La desactivación la hará el usuario manualmente con 'deactivate'