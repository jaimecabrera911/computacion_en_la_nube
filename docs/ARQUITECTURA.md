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

## 3. Modelo de datos (payload de cada punto)

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

## 4. Seguridad cloud

| Medida | Implementación |
|---|---|
| Sin credenciales en el código | `python-dotenv` lee `.env`; el archivo está en `.gitignore`. |
| Sin secretos en las salidas | El notebook y la app muestran `QDRANT_API_KEY` como `[REDACTED]`. |
| Sin endpoint privado publicado | La URL del clúster se redacta en las salidas del notebook. |
| Modo degradado seguro | Sin credenciales, el sistema usa Qdrant in-memory: nada sale del equipo. |
| Credenciales por entorno | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `EMBEDDING_MODEL_NAME` (plantilla en `.env.example`). |

---

## 5. Reproducibilidad

1. `uv sync` resuelve un entorno **determinista** a partir de `uv.lock`.
2. `uv run rag pipeline` ejecuta los 5 pasos y escribe `output/benchmark_summary.json`.
3. `uv run python scripts/execute_notebook.py` reejecuta el notebook y guarda sus outputs.
4. `uv run rag poster` / `summary-pdf` / `diagram` regeneran los entregables visuales
   **consumiendo el mismo JSON de métricas**, por lo que ningún número del repositorio
   está escrito "a mano".
