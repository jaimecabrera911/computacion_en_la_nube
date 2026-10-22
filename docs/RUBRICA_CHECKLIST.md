# ✅ Checklist de la Rúbrica ACA N° 1 — Evidencia de cada criterio

> Mapa directo entre los **6 criterios de la rúbrica del enunciado** y los archivos concretos
> que demuestran su cumplimiento. Úsalo antes de entregar: si algo no está en la lista, o
> el archivo no existe, o el número no coincide, todavía no está listo.

---

## Resumen de pesos

| # | Criterio | Peso | Estado | Evidencia principal |
|---|---|---|---|---|
| 1 | Aprovisionamiento Cloud y Conectividad | 20% | ✅ | `notebooks/…ipynb` §1 · `src/rag/vector_store.py` |
| 2 | Calidad de Datos y Chunking | 20% | ✅ | `notebooks/…ipynb` §2 · `src/rag/document_processor.py` |
| 3 | Generación de Embeddings e Indexación | 20% | ✅ | `notebooks/…ipynb` §3 · `src/rag/vector_store.py` |
| 4 | Validación de Búsqueda Semántica | 15% | ✅ | `notebooks/…ipynb` §4-5 · `docs/RESULTADOS.md` |
| 5 | Póster tipo Infografía (PDF) | 15% | ✅ | `output/Poster_Infografia_ACA1_RAG_Salud.pdf` |
| 6 | Sustentación en Video (3-5 min) | 10% | ✅ | `GUION_SUSTENTACION_VIDEO.md` · `app/app_sustentacion.py` |

---

## 1. Aprovisionamiento Cloud y Conectividad (20%)

| Requisito de la rúbrica | Dónde se cumple | Verificación |
|---|---|---|
| Despliegue y configuración correcta del almacén vectorial en el servicio seleccionado | `src/rag/vector_store.py::create_or_recreate_collection` | Colección `clinical_heart_disease_rag` en el clúster, 349 vectores, dimensión 384. |
| Métrica de distancia apropiada | `src/rag/config.py` (`Distance.COSINE`) | Los embeddings se normalizan en L2 ⇒ el dot product **es** la similitud coseno. |
| Gestión segura de credenciales mediante variables de entorno (`.env`) | `src/rag/config.py` (`python-dotenv`) · `.env.example` · `.gitignore` | Cero API keys en el código; en las salidas del notebook la clave aparece como `[REDACTED]`. |
| Estructura modular del repositorio | `src/rag/` (paquete instalable) | `config · dataset_loader · document_processor · vector_store · evaluation · pipeline · cli · reporting`. |
| Escalabilidad serverless y costo cero | Qdrant Cloud Free Tier 1 GB | 349 vectores ≈ 0,5 MB; margen enorme frente al límite. |

**Complemento exigido por el enunciado (Sección 2):** el dataset base está acompañado de un corpus
propio no estructurado de **3 documentos** (Markdown + PDF) en `data/documents/`. ✅

---

## 2. Calidad de Datos y Chunking (20%)

| Requisito de la rúbrica | Dónde se cumple | Verificación |
|---|---|---|
| Idoneidad del corpus de documentos complementarios | `data/documents/DOC_01..03` | Guías clínicas de MinSalud/OPS, ACC/AHA y ESC/OMS, en Markdown y PDF. |
| Justificación técnica del tamaño de los fragmentos | `src/rag/document_processor.py` (docstring) y §2 del notebook | Tabla que justifica `chunk_size=512` (ventana de 256 wordpiece tokens del encoder) y `chunk_overlap=64` (continuidad entre cortes). |
| Ingesta estructurada reproducible | `src/rag/dataset_loader.py` (`ucimlrepo`, ID 45) | 303 pacientes × 14 variables, descarga programática. |
| Enriquecimiento de metadatos | `src/rag/dataset_loader.py` | `age`, `sex`, `age_group`, `chest_pain_label`, `risk_level`, `disease_present`, `major_vessels`… |
| Narrativización de datos tabulares | `src/rag/dataset_loader.py::patient_row_to_clinical_narrative` | `cp=1` → «Angina típica (Tipo 1, dolor retroesternal opresivo)». |
| Jerarquía de separadores | `src/rag/document_processor.py` | `["\n\n", "\n", ". ", " ", ""]`: párrafos → líneas → oraciones → palabras. |

---

## 3. Generación de Embeddings e Indexación (20%)

| Requisito de la rúbrica | Dónde se cumple | Verificación |
|---|---|---|
| Integración idónea del modelo de representaciones semánticas | `src/rag/config.py` (`EMBEDDING_MODEL_NAME`) | `paraphrase-multilingual-MiniLM-L12-v2` (SBERT multilingüe, CPU local, sin costo de API). |
| Vectorización eficiente | `src/rag/vector_store.py::index_corpus_in_batches` | `model.encode(..., normalize_embeddings=True)` por lotes de 64 textos. |
| Inserción por lotes (batch upsert) sin exceder la capa gratuita | `src/rag/vector_store.py` | `client.upsert(...)` en lotes de 64; 349 vectores en ~20 s. |
| Normalización y coherencia con la métrica | `src/rag/vector_store.py` | Vectores L2 → dot product = coseno. |

---

## 4. Validación de Búsqueda Semántica (15%)

| Requisito de la rúbrica | Dónde se cumple | Verificación |
|---|---|---|
| Consultas en lenguaje natural bien estructuradas | `src/rag/evaluation.py::BENCHMARK_QUERIES` | 5 consultas clínicas: angina, depresión del ST, metas lipídicas/DAPT, cohorte de pacientes, consulta cross-modal. |
| Retorno consistente de los Top-k fragmentos más afines | `notebooks/…ipynb` §4.1-4.2 | Cada consulta muestra su Top-5 con score y metadatos. |
| Análisis cuantitativo de las puntuaciones de similitud coseno | `docs/RESULTADOS.md` · `output/similarity_distribution.png` · `output/metricas_similitud_notebook.png` | Media, mín, máx, desviación por consulta y degradación por rango. |
| Uso de filtros de metadatos | `src/rag/evaluation.py` | Q4 combina `doc_type` + `sex` + `disease_present` + `age ≥ 60`. |
| Medición de latencia | `src/rag/pipeline.py::measure_latency` | Latencia media de búsqueda remota y coste de vectorización local, con calentamiento de conexión. |

---

## 5. Póster tipo Infografía (15%)

| Requisito de la rúbrica | Estado | Verificación |
|---|---|---|
| Objetivo general y situación problema | ✅ | Bloque 1 del póster. |
| Pilar seleccionado | ✅ | Bloque 1: Salud (cardiopatía isquémica). |
| Resumen del corpus documental | ✅ | Tabla de corpus con entidades, formatos y nº de fragmentos **calculado en tiempo de generación**. |
| Diagrama de la arquitectura cloud | ✅ | `output/cloud_rag_architecture.png` embebido en el póster. |
| Stack tecnológico | ✅ | Tabla de stack con justificación por capa. |
| Ejemplos de consultas semánticas | ✅ | Bloque 4: las 5 consultas con su filtro y su Top-1 real. |
| Hipervínculos funcionales | ✅ | Repositorio, documentación, notebook. |
| PDF en alta resolución | ✅ | ReportLab, imágenes a 300 dpi, tamaño A4 de 2 páginas. |
| Normas APA 7 | ✅ | Bloque de referencias con formato APA 7. |
| Consistencia con las métricas reales | ✅ | El póster se genera **leyendo** `output/benchmark_summary.json`; no hay números escritos a mano. |

> ⚠️ **Pendiente de completar por el estudiante:** el enlace al video de sustentación y el enlace
> público del repositorio deben apuntar a las URLs definitivas antes de la entrega.

---

## 6. Sustentación en Video (10%)

| Requisito de la rúbrica | Estado | Verificación |
|---|---|---|
| Justificación del problema | ✅ | `GUION_SUSTENTACION_VIDEO.md` (bloque 0:00-0:45). |
| Explicación del corpus documental | ✅ | Guion (bloque 1:45-2:30) + §2 del notebook. |
| Justificación de las herramientas cloud | ✅ | Guion (bloque 0:45-1:45) + `docs/ARQUITECTURA.md`. |
| Demostración funcional en tiempo real | ✅ | Notebook §4 y la app `app/app_sustentacion.py` (Streamlit) con consultas reales contra el clúster. |
| Duración 3-5 minutos | ✅ | Guion cronometrado: 4 min 15 s. |

> ⚠️ **Pendiente de completar por el estudiante:** grabar el video, subirlo y poner el enlace
> definitivo en `GUION_SUSTENTACION_VIDEO.md` y en el póster.

---

## Directrices de envío y seguridad cloud

| Directriz | Estado | Verificación |
|---|---|---|
| API keys solo desde variables de entorno (`.env`) | ✅ | `.env` en `.gitignore`; `.env.example` como plantilla; sin secretos en el código ni en el notebook. |
| Notebook reproducible de principio a fin, sin errores | ✅ | `uv run python scripts/execute_notebook.py` ejecuta las 22 celdas sin excepciones. |
| Comentarios metodológicos y visualización del Top-k | ✅ | Cada celda está comentada y muestra el Top-k con score y metadatos. |
| Póster en PDF de alta resolución con hipervínculos | ✅ | `output/Poster_Infografia_ACA1_RAG_Salud.pdf` (verificar los enlaces antes de entregar). |
