# 🏛️ Arquitectura Cloud del Sistema RAG

> Documento de referencia técnica. Si necesitas una versión para explicar en 60 segundos,
> mira el diagrama de abajo; si necesitas el detalle, sigue las secciones.

---

## 1. Diagrama de flujo de datos

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CAPA 1 · FUENTES DE INFORMACIÓN (2 modalidades)                          │
│                                                                          │
│  ┌───────────────────────────────┐   ┌────────────────────────────────┐  │
│  │ DATASET ESTRUCTURADO          │   │ CORPUS NO ESTRUCTURADO         │  │
│  │ UCI Heart Disease (ID 45)      │   │ 3 guías clínicas (MD + PDF)    │  │
│  │ 303 pacientes × 14 variables   │   │ MinSalud · ACC/AHA · ESC/OMS   │  │
│  │ descarga: ucimlrepo           │   │ 3 fuentes PDF + 3 Markdown     │  │
│  └───────────────┬───────────────┘   └───────────────┬────────────────┘  │
└──────────────────┼───────────────────────────────────┼──────────────────┘
                   │                                   │
                   ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CAPA 2 · PROCESAMIENTO LOCAL (CPU, costo cero)                          │
│                                                                          │
│  dataset_loader.py            document_processor.py                       │
│  ┌──────────────────────┐     ┌──────────────────────────────────────┐  │
│  │ Narrativización      │     │ RecursiveCharacterTextSplitter       │  │
│  │ cp=1 → "Angina       │     │ chunk_size = 512                     │  │
│  │   típica (Tipo 1)"   │     │ chunk_overlap = 64                   │  │
│  │ + metadatos:         │     │ separators = ["\n\n","\n",". "," "]   │  │
│  │   age, sex,          │     └───────────────────┬──────────────────┘  │
│  │   risk_level,        │                         │                     │
│  │   disease_present    │                         │                     │
│  │ → 303 viñetas        │                         │ 46 fragmentos        │
│  └──────────┬───────────┘                         │                     │
│             └──────────────┬──────────────────────┘                     │
│                            ▼                                            │
│              ┌──────────────────────────┐                              │
│              │ CORPUS CONSOLIDADO       │                              │
│              │ 303 + 46 = 349 fragmentos│                              │
│              └──────────────┬───────────┘                              │
└─────────────────────────────┼────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CAPA 3 · VECTORIZACIÓN (embeddings abiertos, CPU local)                 │
│                                                                          │
│  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2             │
│  • 384 dimensiones por fragmento                                          │
│  • normalización L2 (‖v‖ = 1) ⇒ dot product = similitud coseno           │
│  • batch de 64 textos por pasada (optimiza CPU y red)                    │
└─────────────────────────────┬────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CAPA 4 · ALMACÉN VECTORIAL SERVERLESS (Qdrant Cloud)                    │
│                                                                          │
│  Colección: clinical_heart_disease_rag                                   │
│  • 349 vectores · dimensión 384 · métrica Distance.COSINE                 │
│  • índice HNSW (búsqueda aproximada sub-lineal)                          │
│  • índices de payload: doc_type · sex · disease_present · risk_level     │
│                        patient_id · age                                  │
│  • Free Tier 1 GB · costo $0.00 · escalado serverless                    │
└─────────────────────────────┬────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CAPA 5 · RECUPERACIÓN SEMÁNTICA (RAG)                                   │
│                                                                          │
│  pregunta en lenguaje natural                                            │
│        │                                                                 │
│        ├─► codificación con el mismo modelo (384 dims, L2)                │
│        ├─► query_points(limit=k, query_filter=…)                         │
│        │        └─ filtra por payload (doc_type, sex, age, riesgo…)      │
│        │        └─ busca los k vectores más cercanos por coseno          │
│        ▼                                                                 │
│  Top-k fragmentos con: score de similitud + metadatos + texto            │
│  (listos para alimentar a un LLM en el siguiente paso del proyecto)      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Decisiones de diseño y su justificación

| Decisión | Alternativas descartadas | Justificación |
|---|---|---|
| **Base vectorial: Qdrant Cloud** | Pinecone, Weaviate, Supabase pgvector, ChromaDB | Free Tier permanente de 1 GB, serverless real, índices de payload nativos y dashboard para la defensa. |
| **Embeddings: SBERT multilingüe local** | Gemini, Cohere, Ollama | Costo cero, sin cuota ni rate limit, CPU suficiente, español clínico natively. |
| **Métrica: Cosine** | Euclidiana, producto punto | Los vectores están normalizados ⇒ el producto punto *es* la similitud coseno, pero más barata. La coseno es invariante a la longitud del texto. |
| **Chunking por caracteres (512/64)** | Chunking por tokens, chunking semántico | 512 caracteres ≈ 128 wordpiece tokens ⇒ cabe en la ventana de 256 del encoder sin truncar. El solapamiento evita partir frases. |
| **Narrativización del dataset** | Indexar el CSV crudo | Un modelo semántico no entiende `cp=1`; sí entiende "angina típica (tipo 1)". Además se conservan los campos como payload para filtrar. |
| **Batch upsert de 64** | Inserción uno a uno | Menos viajes de red, menos riesgo de throttling en la capa gratuita, menor uso de memoria. |
| **Índices de payload** | Filtrar sin índice | Con índice, el filtro no obliga a recorrer el corpus: la latencia se mantiene baja. |
| **Capa gratuita de 1 GB** | Cluster pago | El proyecto exige costo cero; 349 vectores × 384 dims ocupan ~0,5 MB. |

---

## 3. Refinamiento del PASO 5: interpretación automática de la consulta

> **Encuadre normativo:** esto **no** agrega un paso al pipeline ni una herramienta al
> ecosistema. Es parte de la *"función de consulta"* que el ACA N° 1 pide en el PASO 5
> (*"captures una pregunta del usuario, la vectorices, ejecutes la búsqueda por similitud y
> recuperes los fragmentos más relevantes"*), y no introduce dependencias, APIs ni costo.

### El problema que resuelve

Un modelo de embeddings **no es un modelo de lenguaje que razona**: es una función
`texto → vector`. No puede comparar números ni seguir instrucciones. Por eso pedir
*"mayor de 60 años"* dentro del texto de una consulta **no** restringe la edad de los
resultados: esa frase solo aporta una dirección semántica, no una condición.

Medido en este proyecto: la similitud media de los pacientes es **plana** respecto a la edad
(0.5656 en 30-49 años, 0.5672 en 50-59, 0.5620 en 60-69, 0.5445 en 70-80) y el mejor score de
la consulta *"paciente masculino mayor de 60 años con enfermedad coronaria confirmada"*
(0.6375) corresponde a un paciente **de 64 años sin enfermedad confirmada**.

### La solución: separar interpretación y similitud

```
pregunta en lenguaje natural
        ↓
[parser determinista]   ← src/rag/query_parser.py (sin red, sin modelo, sin costo)
        ↓  {doc_type, sexo, edad_min/max, enfermedad, riesgo}
base vectorial: búsqueda por similitud CON el filtro aplicado
        ↓
Top-k resultados
```

| Restricción | Se detecta con | Ejemplo |
|---|---|---|
| `doc_type` | palabras clave por tipo de fuente | *"paciente"* → `patient_record`; *"ergometría"* → `diagnostic_protocol`; *"estatinas"* → `pharmacology_protocol`; *"angina"* → `clinical_guideline` |
| `sexo` | términos de género | *"masculino / hombre / varón"* → `Masculino` |
| `edad ≥` | expresiones numéricas | *"mayor de 60"*, *"más de 60"*, *">60"*, *"60 años o más"*, *"a partir de 60"* |
| `edad ≤` | expresiones numéricas | *"menor de 50"*, *"50 años o menos"* |
| `enfermedad` | positivo y negativo (los negativos van primero) | *"con enfermedad coronaria confirmada"* → `True`; *"sin enfermedad"* → `False` |
| `riesgo` | niveles de riesgo | *"riesgo muy alto"* → `Muy Alto / Crítico` |

### Propiedades de diseño

| Propiedad | Detalle |
|---|---|
| **Determinista** | Listas de palabras clave y expresiones regulares. Sin aleatoriedad ni modelo. |
| **Gratuito** | Cero tokens, cero API, cero cuota. La latencia añadida es ~0 ms. |
| **Explicable** | Devuelve *qué* detectó y *por qué*, y se muestra en pantalla antes de aplicar el filtro. |
| **Sin invenciones** | Solo se aplica lo que la consulta **escribe explícitamente**. Un rango contradictorio (`>70` y `<40`) se descarta con una nota. |
| **Agnóstico de tildes y mayúsculas** | Se normaliza el texto (minúsculas, sin tildes, espacios colapsados). |
| **Reversible** | `--no-auto-filter` en la CLI y una casilla en la app lo desactivan. |
| **Los filtros manuales ganan** | Si el usuario fija una restricción en la interfaz, esa prevalece sobre la detectada. |

### Dónde está disponible

| Interfaz | Cómo se usa |
|---|---|
| **CLI** | Automático por defecto: `uv run rag ask "paciente masculino, mayor de 60 años..."` — el sistema muestra `🧠 INTERPRETÉ LA CONSULTA: …` y aplica el filtro. Se desactiva con `--no-auto-filter`. |
| **App web** | Casilla *"🧠 Interpretar la consulta y aplicar los filtros automáticamente"* (activada por defecto), con los chips de lo detectado. |
| **Notebook** | Celda 4.2: análisis de la consulta y comparación lado a lado con/sin filtro. |

### Por qué reglas y no un LLM

En un dominio clínico con vocabulario controlado, un parser determinístico es **más preciso,
auditable y reproducible**, y no depende de la red durante la demostración. Un LLM tiene más
sentido en el paso siguiente del proyecto: **redactar la respuesta** a partir de los fragmentos
recuperados.

---

## 4. Modelo de datos (payload de cada punto)

```jsonc
{
  "text": "…texto completo del fragmento…",   // el contenido que se recupera

  // --- metadatos de documentos clínicos ---
  "doc_type": "clinical_guideline",            // clinical_guideline | diagnostic_protocol
                                               // | pharmacology_protocol | patient_record
  "title": "Guía de Práctica Clínica: …",      // entidad responsable del documento
  "entity": "MinSalud / OPS / OMS",
  "code": "GPC-CARDIO-2024-V2",
  "domain": "cardiology_prevention",
  "filename": "DOC_01_…md",

  // --- metadatos de fragmentación ---
  "chunk_id": 17,                              // identificador global
  "local_chunk_index": 2,                      // posición dentro del documento
  "chunk_char_length": 498,

  // --- metadatos clínicos (solo en viñetas de paciente) ---
  "patient_id": 125,
  "age": 65,
  "age_group": ">=65",
  "sex": "Masculino",
  "chest_pain_type": 1,
  "chest_pain_label": "Angina típica",
  "resting_blood_pressure": 140,
  "cholesterol": 245,
  "fasting_blood_sugar": 1,
  "max_heart_rate": 150,
  "exercise_angina": 0,
  "st_depression": 2.3,
  "st_slope": 2,
  "major_vessels": 2,
  "thal_status": 7,
  "target_disease": 4,
  "disease_present": true,
  "risk_level": "Muy Alto / Crítico"
}
```

---

## 5. Seguridad cloud

| Medida | Implementación |
|---|---|
| Sin credenciales en el código | `python-dotenv` lee `.env`; el archivo está en `.gitignore`. |
| Sin secretos en las salidas | El notebook y la app muestran `QDRANT_API_KEY` como `[REDACTED]`. |
| Sin endpoint privado publicado | La URL del clúster se redacta en las salidas del notebook. |
| Modo degradado seguro | Sin credenciales, el sistema usa Qdrant in-memory: nada sale del equipo. |
| Credenciales por entorno | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `EMBEDDING_MODEL_NAME` (plantilla en `.env.example`). |

---

## 6. Reproducibilidad

1. `uv sync` resuelve un entorno **determinista** a partir de `uv.lock`.
2. `uv run rag pipeline` ejecuta los 5 pasos y escribe `output/benchmark_summary.json`.
3. `uv run python scripts/execute_notebook.py` reejecuta el notebook y guarda sus outputs.
4. `uv run rag poster` / `summary-pdf` / `diagram` regeneran los entregables visuales
   **consumiendo el mismo JSON de métricas**, por lo que ningún número del repositorio
   está escrito "a mano".
