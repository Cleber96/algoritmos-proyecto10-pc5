## **Práctica calificada 5 CC0E5**

### 1. Alcance y Objetivos de Implementación

* **Implementación Funcional y Eficiente:** Desarrolla el núcleo de la estructura o algoritmo elegido de manera funcional y eficiente.
* **Optimización Clave:** Incluye al menos una optimización crucial, basada en la literatura relevante o en variantes que muestren mejoras significativas de rendimiento.
* **Justificación de Elecciones:** Si decides omitir optimizaciones extremas o variantes, justifica claramente por qué quedan fuera del alcance del proyecto. Se espera una exploración que vaya más allá de una implementación básica.

---

### 3. El Repositorio Público (Entrega de Código)

* **Código Fuente Completo:**
    * Debe estar **claro, bien comentado** (especialmente en secciones complejas de la lógica).
    * Debe estar **estructurado modularmente** para facilitar su comprensión y mantenimiento.
    * Los lenguajes permitidos son **Python, C++ o Rust**.
* **Archivo README.md Exhaustivo:**
    * **Descripción del Proyecto:** Incluye una clara descripción y la motivación teórica detrás de la estructura o algoritmo elegido.
    * **Instrucciones de Ejecución:** Proporciona instrucciones claras y precisas para la **compilación y ejecución** del código, detallando todas las **dependencias necesarias** y los comandos paso a paso.
    * **Estructura del Proyecto:** Explica la **organización de archivos y directorios**.
    * **Diseño de API Pública:** Describe el diseño de la API pública de la estructura implementada.
    * **Documentación Detallada de la API:** Especifica **parámetros de entrada, valores de retorno** y el **manejo de excepciones o errores** para cada función pública.
    * **Archivos de Demostración (Drivers/Demos):** Menciona e indica cómo ejecutar los archivos que muestren el **uso avanzado y variado** de la estructura de datos, ilustrando sus capacidades.
    * **Suite de Pruebas y Scripts de Evaluación:** Debe indicar dónde se encuentran y cómo ejecutar la **suite exhaustiva de pruebas unitarias** (buscando **alta cobertura** del código), y los **scripts dedicados para realizar profiling y benchmarking comparativo** de la implementación (ej., midiendo tiempos en distintos tamaños de datos).
* **Suite Exhaustiva de Pruebas Unitarias:**
    * Utiliza un framework de pruebas (ej., Pytest para Python, Google Test para C++).
    * Busca una **alta cobertura del código** para asegurar la robustez y correctitud de la implementación.
* **Scripts de Profiling y Benchmarking:**
    * Proporciona **scripts dedicados** para realizar **profiling** (identificar cuellos de botella) y **benchmarking comparativo** (medir rendimiento bajo diferentes cargas) de tu implementación.

---

### 4. Documentación Adicional (Informe Técnico)

Además del `README.md`, se requiere un **breve informe técnico** en formato PDF (aproximadamente **3-5 páginas**):

* **Teoría Subyacente en Profundidad:**
    * Explica la teoría del concepto central del proyecto en detalle.
    * Incluye **pruebas de correctitud o eficiencia** para los aspectos más críticos, si son relevantes y manejables dentro del alcance.
    * Cita **referencias bibliográficas**.
* **Decisiones de Diseño Importantes:**
    * Discute las alternativas consideradas durante la implementación.
    * Justifica por qué se optó por la solución final para cada decisión clave.
* **Análisis Empírico de Rendimiento:**
    * Presenta un **análisis detallado de la complejidad y el rendimiento observado** a partir de tus benchmarks y profiling.
    * **Compara los resultados con las expectativas teóricas**.
    * Si es posible, incluye una **comparación con otras implementaciones existentes** (ej., de bibliotecas estándar o implementaciones de referencia).
* **Limitaciones y Futuras Extensiones:**
    * Discute las **limitaciones de la implementación actual**.
    * Propone **posibles extensiones o mejoras futuras**.

---

### 5. Presentación (Video y Exposición Oral)

Tendrás una **exposición oral** el **12 de junio** (con posibilidad de un segundo día si es necesario), que se apoyará en una presentación (PowerPoint, PDF, etc.) y un video:

* **Video (Mínimo 7-10 minutos):**
    * **Explicación Teórica Profunda:** Del concepto, incluyendo una **comparativa** con estructuras o algoritmos alternativos y una **justificación clara de su eficiencia** (especialmente en términos de cotas asintóticas).
    * **Descripción Detallada de la Implementación:** Especifica las **invariantes clave** de la estructura de datos o algoritmo y cómo tu implementación las mantiene durante las operaciones.
    * **Recorrido por el Código:** Explica las partes **más complejas o críticas del código**, detallando las **decisiones de diseño** tomadas y por qué se eligieron ciertas aproximaciones.
    * **Demostración en Vivo de la Funcionalidad:**
        * Debe incluir **escenarios típicos** de uso.
        * Debe incluir **escenarios que pongan a prueba la robustez y correctitud** de la estructura (casos límite, errores, etc.).
    * **Discusión de Desafíos:** Aborda los **desafíos encontrados** durante el desarrollo, las **soluciones implementadas** para superarlos.
    * **Trabajos Futuros/Extensiones:** Menciona las **posibles extensiones o mejoras** de la implementación actual.
    * **Análisis de Resultados (Conciso):** Presenta un análisis breve y conciso de los resultados obtenidos del **profiling o los benchmarks**, conectándolos con las expectativas teóricas.
* **Exposición Oral:**
    * Prepárate para **responder preguntas** sobre el diseño del proyecto y los resultados del benchmarking.
    * Se valorará la **claridad al explicar decisiones de diseño** y **resultados experimentales**.
    * Se priorizará el **manejo de los conceptos y códigos vistos en clase**.

---

### 6. Ponderación de Calificaciones

* **Entrega del Repositorio:** El repositorio completo (README, código, pruebas, benchmarking, informe PDF) representa **5 puntos**.
* **Exposición Oral y Preguntas:** La exposición y la capacidad de respuesta a las preguntas representan **15 puntos**.

-----------------------------

### **Propuesta de proyecto 10: Sistema distribuido de búsqueda de similitud vectorial**

- **Área principal:** Distributed Similarity Search Systems usando M-tree y Hashing consistente.
- **Áreas secundarias:** Simulación y optimización.

#### **Descripción detallada**
Este proyecto consiste en construir un prototipo de un **sistema distribuido** para realizar búsquedas de similitud en un gran conjunto de vectores de alta dimensión (e.g., embeddings de IA). 
Se usará un **M-Tree** en cada nodo para una búsqueda local eficiente y **Hashing Consistente** para distribuir los datos de forma escalable entre los nodos. 

#### **Objetivos clave**
1.  **Implementación del M-Tree:** Implementar la estructura M-Tree desde cero (inserción y búsqueda k-NN/rango).
2.  **Implementación de consistent Hashing:** Implementar el anillo de hashing para mapear vectores a nodos.
3.  **Nodo de servicio:** Crear un servicio que contenga un M-Tree y responda a peticiones.
4.  **Orquestador/Gateway:** Crear un punto de entrada que distribuya las peticiones a los nodos y agregue los resultados.
5.  **Simulación distribuida:** Simular el sistema con varios nodos ejecutándose como procesos separados.

#### **Tecnologías y librerías sugeridas**
* **Lenguaje:** Python (con `Flask` o `FastAPI`) o Go/Java.
* **Librerías:** `NumPy` para cálculos vectoriales.

#### **Entregables y cómo enfocar la evaluación**
* **Presentación:** Explicar el problema de la búsqueda de similitud a gran escala y el rol del M-Tree y el Hashing consistente.
* **Video:** Mostrar la arquitectura en acción: iniciar varios nodos, insertar vectores, mostrar cómo se distribuyen y ejecutar una búsqueda.