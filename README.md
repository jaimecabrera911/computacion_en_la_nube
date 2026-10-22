# 🩺 Sistema RAG Cloud Serverless en Salud — ACA N° 1 Computación en la Nube

**Corporación Unificada Nacional de Educación Superior (CUN)** ·
Especialización en Inteligencia Artificial · Escuela de Ingeniería
**Docente:** Hamilton Rivera Flor · **Estudiante:** Jaime Cabrera

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Astral-blueviolet.svg)](https://docs.astral.sh/uv/)
[![Qdrant](https://img.shields.io/badge/Qdrant_Cloud-Serverless-red.svg)](https://qdrant.tech/)
[![Embeddings](https://img.shields.io/badge/Embeddings-SBERT_multilingue-green.svg)](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

---

## 1. ¿Qué es este proyecto?

Implementa la **canalización (pipeline) completa de un sistema RAG en la nube** para el sector
**salud**, abiding the rúbrica del ACA N° 1:

```
Fuentes (dataset tabular + 3 guías clínicas)
   → ingesta y narrativización
   → fragmentación (chunking)
   → embeddings (SBERT multilingüe, 384 dims)
   → indexación vectorial en Qdrant Cloud (serverless)
   → búsqueda semántica Top-k con filtros de metadatos
```

**Pregunta orientadora que responde el proyecto:**

> *¿Cómo estructurar e implementar una canalización de datos no estructurados en la nube mediante
> bases de datos vectoriales serverless y modelos de embeddings abiertos, que permita indexar y
> recuperar semánticamente información contextual especializada con alta precisión, costo cero y
> baja latencia?*

### Pilar seleccionado: Salud 🩺 (cardiopatía isquémica)

| Componente | Detalle |
|---|---|
| **Dataset base (estructurado)** | UCI Heart Disease (ID 45) — 303 pacientes, 14 variables clínicas, descargado con `ucimlrepo` |
| **Corpus complementario (obligatorio, no estructurado)** | 3 guías clínicas en `data/documents/`: MinSalud-OPS-OMS, ACC/AHA y ESC/OMS (Markdown + PDF) |
| **Base de datos vectorial** | Qdrant Cloud (serverless, Free Tier 1 GB) — colección `clinical_heart_disease_rag` |
| **Modelo de embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, CPU local) |
| **Fragmentación** | `RecursiveCharacterTextSplitter` — `chunk_size=512`, `chunk_overlap=64` |
| **Volumen indexado** | 303 viñetas clínicas + 46 fragmentos documentales = **349 vectores** |

> 📖 **Antes de presentar, lee [`docs/EXPLICACION_SUSTENTACION.md`](docs/EXPLICACION_SUSTENTACION.md)**:
> es la guía paso a paso (analogías, preguntas frecuentes, glosario y guion de 4 minutos) que te
> permite explicar el proyecto sin improvisar.

---

## 2. Documentación del proyecto (léela en este orden)

| Documento | Para qué sirve |
|---|---|
| [`docs/EXPLICACION_SUSTENTACION.md`](docs/EXPLICACION_SUSTENTACION.md) | **Cómo explicar el proyecto**: analogía de la biblioteca, decisiones técnicas y por qué, preguntas frecuentes, glosario, guion de sustentación. |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Diagrama de las 5 capas, modelo de datos (payload), decisiones de diseño y medidas de seguridad. |
| [`docs/RESULTADOS.md`](docs/RESULTADOS.md) | Métricas **medidas** del benchmark (se regenera solo: `uv run rag report`). |
| [`docs/RUBRICA_CHECKLIST.md`](docs/RUBRICA_CHECKLIST.md) | Mapa de cada criterio de la rúbrica → archivo que lo evidencia, con los pendientes marcados. |
| [`GUION_SUSTENTACION_VIDEO.md`](GUION_SUSTENTACION_VIDEO.md) | Guion literal minuto a minuto del video de 3-5 minutos. |

---

## 3. Entregables oficiales

| # | Entregable | Archivo | Estado |
|---|---|---|---|
| 1 | **Notebook técnico reproducible** | [`notebooks/heart_disease_rag_qdrant.ipynb`](notebooks/heart_disease_rag_qdrant.ipynb) | 22 celdas, ejecutado de principio a fin con outputs reales |
| 2 | **Póster tipo infografía (PDF)** | [`output/Poster_Infografia_ACA1_RAG_Salud.pdf`](output/Poster_Infografia_ACA1_RAG_Salud.pdf) | 2 páginas A4, diagramado, con métricas medidas y APA 7 |
| 3 | **Video de sustentación (3-5 min)** | Guion en [`GUION_SUSTENTACION_VIDEO.md`](GUION_SUSTENTACION_VIDEO.md) + demo en [`app/app_sustentacion.py`](app/app_sustentacion.py) | Guion cronometrado (4:15); falta grabar y subir el video |
| 4 | **Documentos complementarios** | [`data/documents/`](data/documents/) | 3 guías clínicas en Markdown y PDF |
| 5 | **Diagrama y métricas** | [`output/`](output/) | Diagrama de arquitectura, gráficas de similitud y resumen JSON |

---

## 4. Instalación y ejecución

```bash
# 1. Instalar el entorno (Python 3.12 + dependencias resueltas por uv)
uv sync

# 2. Configurar las credenciales del clúster
cp .env.example .env      # y pegar QDRANT_URL y QDRANT_API_KEY de https://cloud.qdrant.io

# 3. Ejecutar el pipeline completo (PASO 1 → 5) y guardar las métricas
uv run rag pipeline
```

Sin credenciales, el pipeline arranca en **modo local in-memory** y se puede demostrar igual.

### Comandos disponibles

```bash
uv run rag pipeline                 # pipeline completo + métricas (output/benchmark_summary.json)
uv run rag evaluate                 # solo el benchmark semántico
uv run rag ask "¿cuándo está indicada la angioplastia?"   # consulta puntual
uv run rag ask "..." -f patients -k 5                       # con filtro por tipo y Top-k
uv run rag chat                     # consola interactiva
uv run rag queries                  # muestra la batería oficial de consultas
uv run rag poster                   # regenera el póster en PDF
uv run rag summary-pdf              # regenera la guía fácil de resultados
uv run rag diagram                  # regenera el diagrama de arquitectura
uv run rag corpus-pdf               # regenera los PDF del corpus documental
uv run rag report                   # regenera docs/RESULTADOS.md
uv run rag notebook                 # reejecuta el notebook y guarda sus outputs
uv run rag notebook-report          # genera el informe del notebook en PDF y HTML
```

También con `make`: `make setup · make pipeline · make evaluate · make ask "..." · make test · make poster · make demo`.

### Other entry points

```bash
uv run streamlit run app/app_sustentacion.py   # demo web para la defensa (Streamlit)
uv run jupyter lab                              # abrir el notebook
uv run pytest                                   # pruebas automáticas del pipeline
```

### Estructura del repositorio

```
.
├── README.md · Makefile · pyproject.toml
├── ACA_1_Computacion_Nube_RAG.pdf      # enunciado oficial
├── docs/                                # documentación (leer §2)
├── notebooks/heart_disease_rag_qdrant.ipynb
├── src/rag/                             # el código del sistema
│   ├── config.py · dataset_loader.py · document_processor.py
│   ├── vector_store.py · evaluation.py · pipeline.py · cli.py
│   └── reporting/                       # póster, diagrama, guías PDF, informe de resultados
├── app/app_sustentacion.py              # demo web (Streamlit)
├── data/documents/ · data/sources/      # corpus complementario y fuentes oficiales
├── output/                              # artefactos generados (imágenes, PDFs, métricas)
├── scripts/execute_notebook.py          # reejecución del notebook
└── tests/                               # pruebas automáticas
```

---

## 5. Resultados medidos

Ejecutados contra el clúster real de Qdrant Cloud (ver [`docs/RESULTADOS.md`](docs/RESULTADOS.md)
para el detalle completo y reproducible con `uv run rag report`).

| ID | Consulta | Filtro | Top-1 | Fuente recuperada |
|---|---|---|---|---|
| Q1 | Criterios de angina típica y estratificación | `doc_type: clinical_guideline` | **0.6154** | Guía MinSalud/OPS/OMS |
| Q2 | Depresión del ST > 2 mm con pendiente descendente | `doc_type: diagnostic_protocol` | **0.5733** | Protocolo ACC/AHA |
| Q3 | Metas de c-LDL con estatinas y DAPT | `doc_type: pharmacology_protocol` | **0.5764** | Protocolo ESC/OMS |
| Q4 | Paciente masculino > 60 años con cardiopatía | `doc_type` + `sex` + `disease_present` + `age ≥ 60` | **0.7014** | Historia clínica (UCI) |
| Q5 | Isquemia inducible (SPECT) y revascularización | sin filtro (cross-modal) | **0.5676** | Protocolo ACC/AHA |

| Métrica | Valor |
|---|---|
| Similitud coseno media (Top-1) | **0.6068** |
| Similitud coseno máxima (Top-1) | 0.7014 |
| Degradación por rango (1 → 5) | 0.6068 → 0.5603 → 0.5446 → 0.5235 → 0.4955 (monótona ✅) |
| Latencia de búsqueda remota | **~120 ms** (mediana; depende del enlace de red a la región AWS) |
| Vectorización local (SBERT en CPU) | ~15 ms |
| Costo total de la infraestructura | **$0.00 USD** |

> Las cifras de este README, del póster, de la guía de resultados, de la app de sustentación y del
> notebook provienen **todas del mismo archivo** (`output/benchmark_summary.json`), que genera el
> propio pipeline. No hay números escritos a mano.

---

## 6. Seguridad de las credenciales

* Las claves se leen **exclusivamente** del archivo `.env` mediante `python-dotenv`.
* `.env` está en `.gitignore`; `.env.example` es la plantilla que se comparte.
* En el notebook y en la app, la clave se muestra como `[REDACTED]` y la URL del clúster se
  redacta parcialmente.
* Sin credenciales, el sistema opera en modo local in-memory: ningún dato sale del equipo.

---

## 7. Referencias bibliográficas (APA 7)

* American College of Cardiology, & American Heart Association. (2024). *Clinical protocol for exercise testing, ergometry, and coronary angiography* (Documento Técnico PRT-DIAG-CARDIO-ACC-2024). ACC/AHA.
* Chase, H. (2022). *LangChain: Building applications with LLMs through composability and data augmented generation.* https://github.com/langchain-ai/langchain
* European Society of Cardiology, & World Health Organization. (2024). *Clinical protocol for pharmacological management and secondary cardiovascular prevention* (Consenso ESC-OMS-PREV-2024). ESC/WHO.
* Janosi, A., Steinbrunn, R., Pfisterer, M., & Detrano, R. (1988). *UCI Heart Disease Dataset* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease
* Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474. https://arxiv.org/abs/2005.11401
* Ministerio de Salud y Protección Social, & Organización Panamericana de la Salud. (2024). *Guía de práctica clínica: Diagnóstico y estratificación del riesgo en enfermedad arterial coronaria* (GPC-CARDIO-2024-V2). MinSalud/OPS.
* Qdrant Solutions GmbH. (2024). *Qdrant documentation: Vector search engine & cloud database.* https://qdrant.tech/documentation/
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. https://arxiv.org/abs/1908.10084
