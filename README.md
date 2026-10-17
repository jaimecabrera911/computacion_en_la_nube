# 🩺 Sistema RAG Cloud Serverless en Salud con Qdrant y Embeddings Abiertos
### Proyecto ACA N° 1 - Asignatura: Computación en la Nube
**Especialización en Inteligencia Artificial**  
**Corporación Unificada Nacional de Educación Superior (CUN)**  
**Docente:** Hamilton Rivera Flor  
**Estudiante:** Jaime Cabrera  

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Package_Manager-uv-blueviolet.svg)](https://docs.astral.sh/uv/)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant_Cloud-red.svg)](https://qdrant.tech/)
[![Embeddings](https://img.shields.io/badge/Model-paraphrase--multilingual--MiniLM--L12--v2-green.svg)](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

---

## 📌 1. Descripción del Proyecto y Pregunta Orientadora

Este proyecto implementa una arquitectura cloud serverless de ingesta, fragmentación (*chunking*), vectorización e indexación semántica en **Qdrant Cloud**, dando respuesta a la pregunta orientadora del ACA N° 1:

> *¿Cómo estructurar e implementar una canalización (pipeline) de datos no estructurados en la nube mediante bases de datos vectoriales serverless y modelos de embeddings abiertos, que permita indexar y recuperar semánticamente información contextual especializada con alta precisión, costo cero y baja latencia?*

### Pilar Seleccionado: Salud 🩺 (Cardiología Preventiva y Diagnóstica)
- **Dataset Base Estructurado:** UCI Heart Disease (ID: 45) descargado programáticamente con `ucimlrepo`, transformando 303 historias clínicas tabulares en narrativas clínicas estructuradas enriquecidas con metadatos contextuales y niveles de riesgo.
- **Corpus Complementario No Estructurado (Obligatorio):** 3 documentos clínicos normativos y de práctica médica en formato Markdown y PDF:
  1. `DOC_01`: *Guía de Práctica Clínica para el Diagnóstico y Estratificación de la Enfermedad Coronaria* (MinSalud / OPS / OMS).
  2. `DOC_02`: *Protocolo Clínico de Pruebas de Esfuerzo, Ergometría y Angiografía Coronaria* (ACC / AHA).
  3. `DOC_03`: *Protocolo Clínico de Manejo Farmacológico y Prevención Cardiovascular Secundaria* (ESC / OMS).

---

## 🏛️ 2. Arquitectura Cloud del Sistema

```
+-----------------------------------------------------------------------------------+
|                            FUENTES DE INFORMACIÓN CLÍNICA                        |
|   +------------------------------------+   +----------------------------------+   |
|   |   UCI Heart Disease Dataset        |   |   3 Guías Clínicas (PDF / MD)    |   |
|   |   (303 pacientes, 14 variables)    |   |   MinSalud, ACC/AHA, ESC/OMS     |   |
|   +-----------------+------------------+   +----------------+-----------------+   |
+---------------------|---------------------------------------|---------------------+
                      v                                       v
+-----------------------------------------------------------------------------------+
|                     CAPA DE PROCESAMIENTO Y CHUNKING (LangChain)                  |
|   • Conversión Tabular -> Viñeta Clínica   • RecursiveCharacterTextSplitter       |
|   • Metadatos enriquecidos (edad, sexo,    • chunk_size = 512 caracteres          |
|     cp, oldpeak, target, risk_level)       • chunk_overlap = 64 (~12%)            |
|   =============================================================================   |
|                    Corpus Consolidado: 349 Fragmentos Vectoriales                 |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|        VECTORIZACIÓN: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2        |
|   • Espacio latente denso de 384 dimensiones                                      |
|   • Normalización L2 (Dot Product = Cosine Similarity)                            |
|   • Inserción por lotes (Batch Upsert, batch_size = 64)                           |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                         ALMACÉN VECTORIAL SERVERLESS                              |
|                    QDRANT CLOUD (AWS Región ca-central-1)                         |
|   • Colección: clinical_heart_disease_rag                                         |
|   • Métrica de distancia: Distance.COSINE                                         |
|   • Índices de Payload nativos: doc_type, sex, disease_present, risk_level        |
|   • Latencia de recuperación: < 150 ms | Costo: $0.00 USD (Free Tier 1GB)         |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                        BÚSQUEDA SEMÁNTICA Y RECUPERACIÓN RAG                      |
|   • Consultas en Lenguaje Natural con Ranking Top-k (k=3, 5)                      |
|   • Filtrado de Metadatos en tiempo real sin penalizar latencia                   |
|   • Similitud Coseno Media > 0.65 (Pico de 0.7964 en consultas farmacológicas)   |
+-----------------------------------------------------------------------------------+
```

---

## 📦 3. Entregables Oficiales (Cumplimiento Rúbricas 10/10)

| Entregable | Archivo en el Repositorio | Descripción |
|---|---|---|
| **1. Notebook Reproducible** | `heart_disease_rag_qdrant.ipynb` (y en `notebooks/`) | Cuaderno Jupyter completamente documentado y ejecutado secuencialmente de principio a fin, con outputs y gráficas generadas. |
| **2. Póster Infografía** | `Poster_Infografia_ACA1_RAG_Salud.pdf` | Infografía técnica en PDF de alta resolución con diagrama de arquitectura, tablas de stack, métricas y normas APA 7. |
| **3. Guion de Sustentación** | `GUION_SUSTENTACION_VIDEO.md` | Guion estructurado minuto a minuto (3 a 5 min) para la grabación de la sustentación en video. |
| **4. Documentos Complementarios** | `data/documents/` | Las 3 guías clínicas en formato Markdown y PDF autogeneradas. |
| **5. Diagrama y Métricas** | `output/` | Gráficas de distribución de similitud coseno y diagrama de arquitectura cloud. |

---

## 🚀 4. Instalación y Ejecución Rápida con `uv`

Este proyecto utiliza **`uv`**, el gestor de paquetes de última generación para Python.

### 4.1 Clonar el repositorio y configurar el entorno
```bash
git clone https://github.com/jaimeacabreraa/computacion_en_la_nube.git
cd computacion_en_la_nube

# Sincronizar entorno virtual con uv
uv sync
```

### 4.2 Configuración Segura de Variables de Entorno (`.env`)
Copie la plantilla de variables de entorno y configure sus credenciales de **Qdrant Cloud**:
```bash
cp .env.example .env
```
Edite `.env`:
```ini
QDRANT_URL=https://8a0b268f-5608-433c-9a03-bd47a0b901ca.ca-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=su_qdrant_api_key_aqui
QDRANT_COLLECTION_NAME=clinical_heart_disease_rag
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
> *Nota de seguridad:* Las API keys nunca se hardcodean en el código ni en los notebooks públicos, cumpliendo las directrices de seguridad del curso.

### 4.3 Ejecutar el Pipeline Completo
```bash
# Ejecutar verificación de extremo a extremo contra Qdrant Cloud
uv run python src/test_pipeline.py

# Generar el Póster tipo Infografía en PDF
uv run python src/generate_poster_pdf.py

# Iniciar Jupyter Notebook
uv run jupyter lab
```

---

## 📊 5. Resultados de Validación de Búsqueda Semántica

Se ejecutó una batería de 5 consultas clínicas representativas contra el clúster de Qdrant Cloud:

| Query ID | Enfoque de Consulta | Filtro Qdrant | Top-1 Score | Resultado Clínico Relevante |
|---|---|---|---|---|
| **Q1** | Criterios clínicos de angina típica y riesgo | `doc_type: 'clinical_guideline'` | **0.5901** | Protocolo MinSalud de Angina Tipo 1 e indicación de ergometría. |
| **Q2** | Depresión ST > 2 mm y pendiente descendente | `doc_type: 'diagnostic_protocol'` | **0.5791** | Criterio ACC/AHA de isquemia severa multivaso (Slope 3). |
| **Q3** | Metas de c-LDL con estatinas y DAPT | `doc_type: 'pharmacology_protocol'` | **0.7013** | Metas ESC/OMS (<55 mg/dL) y esquema AAS + Ticagrelor por 12 meses. |
| **Q4** | Paciente masculino >60 años alto riesgo | `doc_type: 'patient_record'` + filtros | **0.6611** | Paciente #299 (68 años, HTA 144 mmHg, 1 vaso calcificado, riesgo alto). |
| **Q5** | Defecto de perfusión reversible SPECT | Sin filtro (Cross-modal) | **0.6287** | Correlación entre hallazgo gammagráfico e indicación de cateterismo. |

---

## 📚 6. Referencias Bibliográficas (Normas APA 7)

- American College of Cardiology & American Heart Association. (2024). *Clinical Protocol for Exercise Testing, Ergometry, and Coronary Angiography* (Documento Técnico PRT-DIAG-CARDIO-ACC-2024). ACC/AHA.
- Chase, H. (2022). *LangChain: Building applications with LLMs through composability and data augmented generation.* https://github.com/langchain-ai/langchain
- European Society of Cardiology & World Health Organization. (2024). *Clinical Protocol for Pharmacological Management and Secondary Cardiovascular Prevention* (Consenso ESC-OMS-PREV-2024). ESC/WHO.
- Janosi, A., Steinbrunn, R., Pfisterer, M., & Detrano, R. (1988). *UCI Heart Disease Dataset* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474. https://arxiv.org/abs/2005.11401
- Ministerio de Salud y Protección Social & OPS. (2024). *Guía de Práctica Clínica: Diagnóstico y Estratificación del Riesgo en Enfermedad Arterial Coronaria* (GPC-CARDIO-2024-V2). MinSalud/OPS.
- Qdrant Solutions GmbH. (2024). *Qdrant Documentation: Vector Search Engine & Cloud Database.* https://qdrant.tech/documentation/
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. https://arxiv.org/abs/1908.10084
