# 📖 Guía de Explicación del Proyecto — Cómo explicar el ACA N° 1 en 10 minutos

> **Propósito de este documento:** que puedas explicar el proyecto de forma clara, ordenada y
> defendible, incluso si alguien te pregunta algo que no está en el código. Está pensado para
> leerse **antes** de la sustentación y para ser tu guion de apoyo si improvisas.

---

## 0. La idea en una sola frase

> **Tomamos historias clínicas y guías médicas, las convertimos en vectores (números que el
> computador sabe comparar por significado) y las guardamos en una base de datos en la nube.
> Después, cuando alguien escribe una pregunta en lenguaje normal, el sistema devuelve los
> fragmentos que realmente responden esa pregunta — en menos de 150 milisegundos y sin pagar un peso.**

---

## 1. El problema y la pregunta orientadora

**Situación problema (en palabras simples):**
La documentación médica está fragmentada: hay datasets tabulares (303 pacientes con 14 variables
numéricas) y hay textos clínicos (guías de práctica clínica en PDF). Un buscador clásico solo
busca palabras exactas: si el paciente escribe *"dolor en el pecho"*, no encuentra nada que diga
*"angina típica"*. Un médico pierde tiempo porque el computador no entiende el **significado**.

**Pregunta orientadora (del enunciado):**
> *¿Cómo estructurar e implementar una canalización (pipeline) de datos no estructurados en la nube
> mediante bases de datos vectoriales serverless y modelos de embeddings abiertos, que permita
> indexar y recuperar semánticamente información contextual especializada con alta precisión,
> costo cero y baja latencia?*

**Nuestra respuesta (objetivo general):**
Implementamos esa canalización completa sobre **Qdrant Cloud** (serverless, capa gratuita) con el
modelo de embeddings abierto **SBERT multilingüe** (`paraphrase-multilingual-MiniLM-L12-v2`),
combinando un dataset estructurado (UCI Heart Disease) con un corpus documental no estructurado
(3 guías clínicas) en una **misma colección vectorial**, y lo validamos con 5 consultas en lenguaje
natural medidas cuantitativamente.

---

## 2. La analogía que lo explica todo (úsala si te pierdes)

Imagina una **biblioteca gigante** donde los libros no se ordenan por título, sino por **contenido**:

| Concepto RAG | Analogía de la biblioteca |
|---|---|
| **Documento** (guía clínica, historia de paciente) | Un libro |
| **Chunk** (fragmento de 512 caracteres) | Un párrafo |
| **Embedding** (vector de 384 números) | La "ficha temática" de ese párrafo, escrita como 384 números |
| **Base vectorial** (Qdrant Cloud) | El catálogo inteligente de la biblioteca |
| **Query** (pregunta del usuario) | Tu pregunta al bibliotecario |
| **Top-k** | Los 3 o 5 párrafos que el bibliotecario te entrega |
| **Payload / metadatos** | La etiqueta de estante: "capítulo 3", "página 45", "editorial X" |
| **Similitud coseno** | Qué tanto coincide la "ficha temática" de tu pregunta con la del párrafo |

Cuando preguntas *"¿qué hago si el ST baja más de 2 mm?"*, el sistema **no busca esas palabras**:
codifica tu pregunta en una ficha temática y busca los párrafos cuya ficha temática sea más
parecida. Por eso encuentra *"depresión del ST > 2.0 mm con pendiente descendente"*, aunque no
coincida ni una palabra.

---

## 3. Las 5 piezas del sistema (memorízalas)

| # | Pieza | Qué hace | Dónde vive en el código |
|---|---|---|---|
| 1 | **Ingesta** | Descarga el dataset UCI y lee los 3 documentos Markdown/PDF | `src/rag/dataset_loader.py`, `src/rag/document_processor.py` |
| 2 | **Narrativización + Chunking** | Convierte cada fila tabular en una viñeta clínica y corta los textos en fragmentos de 512 caracteres con 64 de solapamiento | `src/rag/dataset_loader.py`, `src/rag/document_processor.py` |
| 3 | **Embeddings** | Convierte cada fragmento en un vector denso de 384 dimensiones (SBERT multilingüe, normalizado) | `src/rag/vector_store.py` |
| 4 | **Almacén vectorial cloud** | Sube los 349 vectores a la colección `clinical_heart_disease_rag` en Qdrant Cloud con métrica coseno e índices de payload | `src/rag/vector_store.py` |
| 5 | **Búsqueda semántica** | Codifica la pregunta y devuelve el Top-k más afín, con filtros de metadatos | `src/rag/vector_store.py`, `src/rag/evaluation.py` |

> **Dato clave para memorizar:** `303 viñetas clínicas + 46 fragmentos documentales = 349 vectores`
> en **una sola colección**. Esa es la "integración multimodal" del proyecto.

---

## 4. Las decisiones técnicas y **por qué** se tomaron (preguntas más frecuentes)

| Pregunta que te pueden hacer | Respuesta |
|---|---|
| **¿Por qué Qdrant y no Pinecone o Chroma?** | La rúbrica lo deja a elección. Qdrant Cloud ofrece **Free Tier permanente de 1 GB**, es **serverless** (escala a cero, sin servidor que pagar), y tiene **índices de payload nativos**, que es justo lo que necesitamos para filtrar por `doc_type`, `sex` o `age` sin penalizar la latencia. |
| **¿Por qué un modelo local y no una API de embeddings?** | Costo cero real, sin cuota ni rate limit, y funciona en CPU. `paraphrase-multilingual-MiniLM-L12-v2` está entrenado en 50+ idiomas, así que entiende español y terminología clínica sin necesidad de translated queries. |
| **¿Por qué 384 dimensiones y no 768 o 1536?** | Es la dimensión nativa del modelo elegido. Más dimensiones = más memoria y búsqueda más lenta. 384 con normalización L2 da un excelente equilibrio entre cobertura semántica y costo cero en la capa gratuita. |
| **¿Por qué coseno y no producto punto o Euclidiana?** | Los vectores están normalizados (L2 = 1), así que el **dot product es exactamente la similitud coseno**, pero con una operación más barata. La coseno mide el **ángulo** entre vectores: dos textos que hablan de lo mismo tienen ángulos pequeños, sin importar su longitud. |
| **¿Por qué `chunk_size=512`?** | Es el límite que impone el modelo: su encoder trunca a 256 *wordpiece tokens*. Con 512 caracteres (~128 tokens) el fragmento **cabe completo**, así que nunca se pierde contenido por truncamiento y el fragmento es semánticamente autocontenido. |
| **¿Por qué `chunk_overlap=64`?** | Sin solapamiento, una frase que cae justo en el corte se parte en dos fragmentos incomprensibles. Con 64 caracteres (~12,5 %) de solapamiento se conserva la continuidad del texto entre fragmentos. |
| **¿Qué son los índices de payload?** | Qdrant no solo guarda el vector: guarda también metadatos (tipo de documento, edad, sexo, riesgo). Indexarlos es crear "ficheros" para que el filtro sea tan rápido como la búsqueda vectorial. Sin índice, un filtro obliga a recorrer todo el corpus. |
| **¿Por qué narrativizar el dataset tabular?** | Un RAG solo funciona con **texto**. `cp = 1` no significa nada para un modelo de lenguaje; *"Angina típica (Tipo 1, dolor retroesternal opresivo)"` sí. Además, al narrativizar conservamos los campos clave como metadatos, que es lo que habilita los filtros clínicos. |
| **¿Cómo se manejan las credenciales?** | `python-dotenv` las lee **solo** del archivo `.env`, que está en `.gitignore`. En el notebook y en el código no hay ni una API key escrita. En las salidas del cuaderno la clave se muestra como `[REDACTED]`. |
| **¿Y si no tengo credenciales?** | El pipeline detecta su ausencia y arranca en **modo local in-memory** (`QdrantClient(location=":memory:")`): el proyecto se puede ejecutar y demostrar sin conexión, aunque sin persistencia. |
| **¿Qué tan bueno es el sistema?** | Ver `docs/RESULTADOS.md`. Resumen: similitud coseno media Top-1 de **0.61**, con degradación monótona por rango (Rank 1 > Rank 2 > … > Rank 5) y latencia remota de **~120 ms** con conexión establecida (vectorización local: ~15 ms). |
| **¿Qué es lo que NO hace?** | No genera texto con un LLM: es la capa de **recuperación** (que es lo que pide el ACA 1). El paso siguiente sería feeding de esos fragmentos a un LLM para redactar la respuesta con citas. |

---

## 5. Recorrido guiado del repositorio (mapa para no perderse)

```
computacion_en_la_nube/
├── README.md                  ← punto de entrada: qué es, cómo se ejecuta, qué se entrega
├── pyproject.toml             ← dependencias y entry point `rag`
├── Makefile                   ← atajos: make setup, make pipeline, make test...
├── ACA_1_Computacion_Nube_RAG.pdf   ← enunciado oficial (referencia)
│
├── docs/                      ← LA PARTE QUE HAY QUE LEER PARA EXPLICAR
│   ├── EXPLICACION_SUSTENTACION.md  ← este documento
│   ├── ARQUITECTURA.md              ← diagrama y flujo de datos
│   ├── RESULTADOS.md                ← métricas medidas (única fuente de verdad)
│   └── RUBRICA_CHECKLIST.md         ← cada criterio de la rúbrica y su evidencia
│
├── notebooks/
│   └── heart_disease_rag_qdrant.ipynb  ← ENTREGABLE 1: notebook reproducible y ejecutado
│
├── src/rag/                   ← el código del sistema (paquete instalable)
│   ├── config.py              ← credenciales, modelo, chunking, rutas
│   ├── dataset_loader.py      ← ingesta + narrativización del dataset UCI
│   ├── document_processor.py  ← carga y fragmentación del corpus documental
│   ├── vector_store.py        ← cliente Qdrant, indexación, búsqueda semántica
│   ├── evaluation.py          ← batería oficial de consultas + métricas
│   ├── pipeline.py            ← orquestador PASO 1 → PASO 5
│   ├── cli.py                 ← interfaz de línea de comandos (`rag ...`)
│   └── reporting/             ← póster, diagrama y guías PDF
│
├── app/app_sustentacion.py   ← demo web (Streamlit) para la defensa
├── data/
│   ├── documents/             ← corpus complementario (3 .md + 3 .pdf)
│   └── sources/               ← PDFs oficiales de referencia
│
├── output/                    ← artefactos generados (imágenes, métricas, PDFs)
├── scripts/execute_notebook.py ← reejecuta el notebook con outputs
├── tests/                     ← pruebas automáticas del pipeline
└── GUION_SUSTENTACION_VIDEO.md  ← guion minuto a minuto del video (3-5 min)
```

---

## 6. Cómo ejecutar el proyecto (memoriza estos 3 comandos)

```bash
uv sync                     # 1. Instala el entorno (una sola vez)
uv run rag pipeline         # 2. Ejecuta TODO: ingesta → chunking → embeddings → indexación → evaluación
uv run rag ask "¿cuándo está indicada la angioplastia?"   # 3. Pregunta al sistema
```

Atajos equivalentes: `make setup`, `make pipeline`, `make ask`, `make test`, `make poster`.

---

## 7. Checklist de la sustentación (3-5 minutos)

| Tiempo | Bloque | Qué dices | Qué muestras |
|---|---|---|---|
| 0:00-0:45 | **Problema** | La pregunta orientadora y por qué un buscador por palabras no sirve en medicina | Portada del póster |
| 0:45-1:45 | **Arquitectura y stack** | Las 4 capas: fuentes → procesamiento → embeddings → Qdrant Cloud; credenciales en `.env` | Diagrama de arquitectura + consola de Qdrant Cloud |
| 1:45-2:30 | **Datos y chunking** | 303 viñetas clínicas + 46 fragmentos documentales = 349 vectores; justifica 512/64 | Sección 2 del notebook |
| 2:30-3:45 | **Demo en vivo** | 2-3 consultas reales con sus scores y el Top-k recuperado | Notebook (celdas 15-17) o la app Streamlit |
| 3:45-4:15 | **Métricas y cierre** | Similitud media 0.61, degradación monótona, latencia ~120 ms, costo cero | Gráfica de métricas + póster |

Los **números exactos y verificados** están en `docs/RESULTADOS.md` y en
`output/benchmark_summary.json` (generado por el propio pipeline).

---

## 8. Caso de demostración estrella: por qué el filtro importa más que el score

Esta es **la prueba más potente para la defensa**, porque contrasta dos ejecuciones de la misma
consulta y muestra que el sistema no solo "busca texto", sino que **aplica criterios clínicos**.

**Consulta:** `paciente masculino, mayor de 60 años, con enfermedad coronaria confirmada`

### A) Con filtro simple (solo `doc_type = patient_record`)

| Rank | Score | Paciente | Edad | ¿Enfermedad confirmada? | Cumple el perfil pedido |
|---|---|---|---|---|---|
| 1 | 0.6375 | #21 | 64 | ❌ no | parcial |
| 2 | 0.6371 | #42 | 40 | ❌ no | ❌ |
| 3 | 0.6369 | #299 | 45 | ✅ sí | ❌ (45 años) |
| 4 | 0.6202 | #275 | 59 | ✅ sí | ❌ (59 años) |
| 5 | 0.6163 | #15 | 52 | ❌ no | ❌ |

> ⚠️ **El score más alto (0.6375) corresponde a un paciente que NO cumple los criterios.**
> El buscador devolvió lo *más parecido en texto*, no lo *clínicamente correcto*.

### B) Con filtro compuesto (`sex` + `disease_present` + `age ≥ 60`)

| Rank | Score | Paciente | Edad | ¿Enfermedad confirmada? | Cumple el perfil pedido |
|---|---|---|---|---|---|
| 1 | 0.6115 | #244 | 61 | ✅ sí | ✅ |
| 2 | 0.5993 | #25 | 60 | ✅ sí | ✅ |
| 3 | 0.5976 | #188 | 66 | ✅ sí | ✅ |
| 4 | 0.5950 | #246 | 67 | ✅ sí | ✅ |
| 5 | 0.5943 | #72 | 67 | ✅ sí | ✅ |

### Las tres conclusiones que puedes decir en voz alta

1. **El score mide similitud semántica, no veracidad clínica.** Con el filtro compuesto los scores
   son *más bajos* (0.6115 vs 0.6375) y sin embargo **las respuestas son correctas**. La similitud
   mide qué tan parecidas son dos frases; el filtro decide quién es un candidato válido.
2. **Los índices de payload convierten una búsqueda en una consulta clínica.** Cuatro condiciones
   (tipo de documento, sexo, enfermedad confirmada y edad) se aplican en la **misma pasada** del
   índice HNSW, sin recorrer el corpus completo.
3. **El filtro además mejora la latencia:** ~204 ms con filtro compuesto frente a ~595 ms con el
   filtro simple, porque el conjunto candidato se reduce antes de ordenar por similitud.

> En toda la colección existen **39 pacientes** que cumplen exactamente ese perfil (hombres, con
> enfermedad confirmada, ≥ 60 años), con edades entre 60 y 77 años. Subiendo `--top-k` se puede
> recorrer el grupo completo.

### Cómo reproducirlo

```bash
uv run rag ask "paciente masculino, mayor de 60 años, con enfermedad coronaria confirmada" \
  -f patients --sex Masculino --min-age 60 --disease true -k 5
```

---

## 9. Glosario rápido (por si te preguntan un término)

| Término | Definición en una línea |
|---|---|
| **RAG** | *Retrieval-Augmented Generation*: en vez de inventar la respuesta, el modelo primero **recupera** documentos relevantes y se apoya en ellos. |
| **Embedding** | Representación numérica de un texto que captura su significado; textos parecidos → vectores parecidos. |
| **Vector** | Lista de números (aquí 384) donde la distancia representa la similitud semántica. |
| **Chunk / Fragmento** | Pedazo de documento (aquí 512 caracteres) que se indexa como unidad atómica. |
| **Chunking** | Proceso de dividir documentos largos en fragmentos manejables. |
| **Embedding dim. / solapamiento** | 384 dimensiones por vector; 64 caracteres de solapamiento entre fragmentos contiguos. |
| **HNSW** | Índice aproximado de búsqueda vectorial (grafos jerárquicos) que permite buscar en log(n) en vez de recorrer todo. |
| **Métrica de distancia** | La fórmula que decide qué vector es "más cercano": aquí **coseno** (ángulo). |
| **Payload** | Los metadatos guardados junto a cada vector (`doc_type`, `age`, `sex`, `risk_level`…). |
| **Índice de payload** | Estructura que hace los filtros tan rápidos como la búsqueda vectorial. |
| **Top-k** | Cuántos fragmentos se devuelven (aquí k = 3 a 5). |
| **Batch upsert** | Insertar muchos vectores en una sola llamada de red, en vez de uno por uno. |
| **Serverless** | Infraestructura gestionida que escala a cero cuando no se usa: se paga solo por lo que se consume (aquí, nada). |
| **Free Tier** | Capa gratuita del proveedor: 1 GB en Qdrant Cloud. |
| **Normalización L2** | Dividir cada vector por su longitud para que todos midan 1; así el producto punto = coseno. |
| **ucimlrepo** | Librería oficial de Python para descargar datasets de UCI de forma programática y reproducible. |
| **Glosario de sinónimos** | Un mismo concepto dicho de varias formas ("IAM", "infarto de miocardio agudo"). |

---

## 10. Si te preguntan "¿qué haría distinto?"

Respuestas honestas y sólidas (valoran la madurez del proyecto):

1. **Usar un modelo fine-tunedo en el dominio** (`PubMedBERT`, `MedCPT`) en lugar de un multilingüe
   general: subiría la similitud en consultas cardiológicas muy especializadas.
2. **Chunking semántico** (por ejemplo, con un modelo de MMR o partición por títulos de sección en
   lugar de tamaño fijo) para respetar mejor la estructura de las guías clínicas.
3. **Re-ranking** con un cross-encoder sobre los 50 candidatos iniciales para mejorar el orden del Top-k.
4. **Evaluar con un ground truth** etiquetado por un cardiólogo (recall@k, MRR, nDCG) en lugar de
   solo scores de similitud.
5. **Cerrar el lazo RAG** con un LLM que redacte la respuesta a partir de los fragmentos recuperados,
   con cita a la fuente.
6. **Versionar la colección** y desplegar con CI/CD para que cada commit reindexe y valide el
   pipeline automáticamente.

---

## 11. Documentos relacionados

| Documento | Para qué te sirve |
|---|---|
| `docs/ARQUITECTURA.md` | Explicar el diagrama de arquitectura con detalle técnico. |
| `docs/RESULTADOS.md` | Citar las métricas exactas sin inventar números. |
| `docs/RUBRICA_CHECKLIST.md` | Saber dónde está la evidencia de cada criterio de la rúbrica. |
| `GUION_SUSTENTACION_VIDEO.md` | Guion literal minuto a minuto del video (4:15). |
| `GUION_3_MINUTOS.md` | Versión corta de 3 minutos, en lenguaje sencillo, lista para leer tal cual. |
| `output/benchmark_summary.json` | Fuente de verdad de los números (la genera el pipeline). |
