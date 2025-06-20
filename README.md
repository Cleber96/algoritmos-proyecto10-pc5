```markdown
distributed_vector_search/
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── models.py                   # Definiciones de clases para Vector, Resultado de Búsqueda, etc.
│   │   └── utils.py                    # Funciones de utilidad (ej. cálculo de distancias, logging)
│   │
│   ├── m_tree/
│   │   ├── __init__.py
│   │   ├── m_tree.py                   # Implementación de la estructura de datos M-Tree
│   │   └── node.py                     # Definición de los nodos internos y hojas del M-Tree
│   │
│   ├── consistent_hashing/
│   │   ├── __init__.py
│   │   └── consistent_hasher.py        # Implementación del anillo de Hashing Consistente
│   │
│   ├── worker_node/
│   │   ├── __init__.py
│   │   ├── app.py                      # Aplicación Flask/FastAPI para el nodo worker
│   │   └── worker_service.py           # Lógica de negocio del worker (interacción con M-Tree)
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── app.py                      # Aplicación Flask/FastAPI para el orquestador/gateway
│   │   └── orchestrator_service.py     # Lógica de negocio del orquestador (enrutamiento, agregación)
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── run_simulation.py           # Script principal para iniciar y coordinar la simulación
│   │   └── data_generator.py           # Script para generar vectores de prueba
│   │
│   ├── benchmarking/                 # ⬅ NUEVO: scripts de perfilado y benchmarks
│   │   ├── __init__.py
│   │   ├── benchmark_local.py        # M-Tree vs. faiss-flat, sklearn-KDTree, etc.
│   │   └── benchmark_distributed.py  # Latencia extremo-a-extremo con tamaños crecientes
│   │
│   └── profiling/
│       ├── __init__.py
│       ├── profile_worker.py         # cProfile + snakeviz
│       └── profile_mt_insert.py      # Hot-spots de inserción masiva
│
├── notebooks/
│   ├── f  # Contexto del problema y conceptos iniciales
│   ├── 01_M-Tree_Implementacion.ipynb          # Desarrollo y pruebas del M-Tree
│   ├── 02_Hashing_Consistente_Implementacion.ipynb # Desarrollo y pruebas del Consistent Hashing
│   ├── 03_Nodo_Worker_Local.ipynb              # Creación del servicio de Worker (instancia única)
│   ├── 04_Orquestador_y_Comunicacion.ipynb     # Creación del Orquestador y comunicación con Workers
│   ├── 05_Sistema_Distribuido_Completo.ipynb   # Puesta en marcha de la simulación distribuida
│   └── 06_Analisis_y_Optimización.ipynb        # Evaluación de rendimiento y posibles mejoras
│
├── demos/
│   ├── demo_m_tree_local.py                # Demostración autónoma de M-Tree (inserción, k-NN)
│   ├── demo_consistent_hashing.py          # Demostración autónoma de Consistent Hashing (distribución, rebalanceo)
│   ├── demo_single_worker_api.py           # Demostración autónoma de la API de un solo Worker
│   ├── demo_orchestrator_interaction.py    # Demostración autónoma de Orquestador con 2-3 Workers
│   └── demo_full_system_run.py             # Demostración autónoma de la simulación completa
│
├── tests/
│   ├── __init__.py
│   ├── test_m_tree.py                  # Pruebas unitarias para la implementación del M-Tree
│   ├── test_consistent_hashing.py      # Pruebas unitarias para el Hashing Consistente
│   ├── test_worker_node.py             # Pruebas de integración para el nodo worker
│   └── test_orchestrator.py            # Pruebas de integración para el orquestador
│
├── docs/                             # ⬅ NUEVO
│   ├── report.pdf                    # Informe técnico final (3–5 págs.)
│   ├── report_src.tex                # Fuente LaTeX/Markdown
│   ├── slides.pdf                    # Diapositivas para la exposición
│   └── slides_src/                   # Fuente (PowerPoint, Beamer o Reveal.js)
│
├── data/
│   └── sample_vectors.json             # Ejemplo de archivo de datos de entrada (opcional)
│
├── config/
│   └── settings.py                     # Archivo de configuración (puertos, direcciones, etc.)
│
├── scripts/
│   ├── start_workers.sh                # Script para iniciar múltiples nodos worker
│   ├── start_orchestrator.sh           # Script para iniciar el orquestador
│   ├── run_all.sh                      # Script para iniciar todo el sistema simulado
│   └── create_env.sh                 # ⬅ NUEVO: instala venv + requirements
│
├── pyproject.toml                    # ⬅ NUEVO: empaquetado + black/ruff/pydantic
├── README.md                           # Descripción del proyecto, cómo ejecutarlo, etc.
├── requirements.txt                    # Dependencias del proyecto (pip install -r requirements.txt)
├── LICENSE                           # Buenas prácticas OSS
└── .gitignore                          # Archivo para ignorar en Git
```