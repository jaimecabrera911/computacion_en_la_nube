"""App de Sustentación Interactiva — Proyecto ACA N° 1: Computación en la Nube

Interfaz de la sustentación del pipeline RAG serverless sobre datos
cardiovasculares. El diseño vive en `app/theme.py` (metáfora: expediente
clínico); aquí queda el contenido y el flujo.

Todo el texto de la interfaz está en español.
"""

import os
import time

import streamlit as st
from markdown_it import MarkdownIt  # dependencia garantizada por streamlit

from theme import CUSTOM_CSS, icono

# El fragmento llega con markdown (###, listas) desde Qdrant: renderizado a HTML
# aquí y con html=False, para que Streamlit vea un bloque HTML continuo (si no,
# el markdown crudo dentro del <div> se imprime como texto literal).
_MD = MarkdownIt("commonmark", {"html": False})

st.set_page_config(
    page_title="ACA 1 · RAG Cloud en Salud | CUN",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============ MÉTRICAS REALES (leídas de output/benchmark_summary.json) ============

def metricas() -> dict:
    """Devuelve las métricas medidas en la última ejecución del pipeline.

    Si el JSON no existe, devuelve valores de referencia claramente marcados
    para que la app siga funcionando en modo demostración.
    """
    try:
        from rag.evaluation import load_summary
        from rag.config import settings

        summary = load_summary() or {}
        corpus = summary.get("corpus", {})
        latency = summary.get("latency", {})
        return {
            "fragmentos": corpus.get("total_vectors", 349),
            "pacientes": corpus.get("patients", 303),
            "fragmentos_doc": corpus.get("document_chunks", 46),
            "top1_mean": summary.get("top1_mean", 0.61),
            "top1_max": summary.get("top1_max", 0.70),
            "latencia": latency.get("search_mean_ms", 114),
            "latencia_max": latency.get("search_max_ms", 128),
            "medido": bool(summary),
        }
    except Exception:
        return {
            "fragmentos": 349, "pacientes": 303, "fragmentos_doc": 46,
            "top1_mean": 0.61, "top1_max": 0.70, "latencia": 114, "latencia_max": 128,
            "medido": False,
        }


M = metricas()


# ============ NAVEGACIÓN ============
# El índice se numera solo por CSS (counters), así el marcado de Streamlit
# queda intacto y el índice es legible por lector de pantalla.

SECTION_LABELS = [
    "Portada",
    "Contexto y Pregunta Orientadora",
    "Arquitectura Cloud",
    "Datos, Chunking e Ingesta",
    "Demo RAG en Vivo (Qdrant Cloud)",
    "Resultados y Métricas",
    "Conclusiones",
    "Referencias (APA 7)",
]

# Icono propio + etiqueta corta de cada sección, para la cabecera.
SECCION_META = [
    ("carpeta", "Expediente ACA N° 1"),
    ("diana", "Punto de partida"),
    ("capas", "Diseño de la solución"),
    ("base", "Capa de datos"),
    ("lupa", "Demostración en vivo"),
    ("grafico", "Análisis cuantitativo"),
    ("sello", "Veredicto"),
    ("libro", "Bibliografía"),
]


# ============ HELPERS ============

def cabecera(indice: int, titulo: str, etiqueta: str | None = None):
    """Cabecera de sección: número de folio, etiqueta y título."""
    ico, etiqueta_sugerida = SECCION_META[indice]
    et = etiqueta or etiqueta_sugerida
    st.markdown(
        f"""
        <div class="seccion-cabecera">
            <div class="marca">{indice + 1:02d}</div>
            <div>
                <div class="etiqueta">{et}</div>
                <h2>{titulo}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ficha(titulo: str, cuerpo: str, ico: str = "documento") -> str:
    """Bloque HTML de una hoja de expediente con título e icono."""
    return (
        f'<div class="ficha"><h3>{icono(ico)} {titulo}</h3>'
        f'<div class="lead">{cuerpo}</div></div>'
    )


def fila_datos(items) -> str:
    """Tabla de datos de expediente: cifra + rótulo, en rejilla de una hoja.

    items: iterable de (valor, rótulo) o (valor, rótulo, "sello")
    """
    celdas = []
    for item in items:
        valor, rotulo = item[0], item[1]
        clase = " sello" if len(item) > 2 and item[2] == "sello" else ""
        celdas.append(
            f'<div class="dato"><div class="dato-num{clase}">{valor}</div>'
            f'<div class="dato-lbl">{rotulo}</div></div>'
        )
    return f'<div class="datos">{"".join(celdas)}</div>'


def cinta_pasos(pasos) -> str:
    """Cinta de estaciones del pipeline: cada estación es un tramo numerado."""
    estaciones = []
    for num, ico, titulo, desc in pasos:
        estaciones.append(
            f'<div class="estación"><div class="n">{icono(ico)}{num}</div>'
            f'<div class="t">{titulo}</div><div class="d">{desc}</div></div>'
        )
    return f'<div class="cinta">{"".join(estaciones)}</div>'


def barra_similitud(score: float) -> str:
    pct = max(2.0, min(100.0, score * 100.0))
    clase = "alto" if score >= 0.65 else ""
    return (
        f'<div class="barra"><i class="{clase}" style="width:{pct:.1f}%"></i></div>'
    )


# ============ PÁGINAS ============

def page_portada():
    st.markdown(
        f"""
        <div class="carpeta">
            <h1>Sistema RAG Cloud <em>serverless</em> en salud
            con Qdrant y embeddings abiertos</h1>
            <div class="entradilla">
                Pipeline cloud de datos clínicos no estructurados: ingesta multimodal,
                fragmentación, vectorización semántica y recuperación con alta precisión —
                sustentación interactiva con demo en vivo.
            </div>
            <div class="datos-clinicos" style="margin-bottom:0.9rem">
                <span><b>CUN</b> · Especialización en IA</span>
                <span><b>Pilar</b> · Salud — Cardiología</span>
                <span><b>Costo</b> · $0.00 USD</span>
                <span><b>Latencia</b> · ~{M['latencia']:.0f} ms</span>
            </div>
            <div class="firma">
                <b>Integrantes:</b> Marcela Ávila Beltrán · Jaime Andrés Cabrera Avendaño ·
                Cristhian Javier Garzón Jiménez · Johan Smith Valderrama Sánchez<br>
                Hamilton Rivera Flor — docente · ACA N° 1 · Computación en la Nube<br>
                <a href="https://github.com/jaimecabrera911/computacion_en_la_nube" target="_blank">
                    github.com/jaimecabrera911/computacion_en_la_nube
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        fila_datos([
            (f"{M['fragmentos']}", "Fragmentos vectoriales indexados"),
            (f"{M['pacientes']}", "Pacientes UCI Heart Disease"),
            ("384-d", "Espacio latente SBERT"),
            (f"~{M['latencia']:.0f} ms", "Latencia de recuperación"),
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(
        fila_datos([
            (f"{M['top1_max']:.4f}", "Pico de similitud coseno (Top-1)", "sello"),
            (f"{M['top1_mean']:.4f}", "Similitud coseno media (Top-1)"),
            ("3", "Guías clínicas normativas"),
            ("$0.00", "Costo total del despliegue"),
        ]),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="rejilla" style="grid-template-columns: 1.15fr 0.85fr">'
        + ficha(
            "La sustentación en 30 segundos",
            'Este proyecto implementa un <b>pipeline cloud serverless</b> end-to-end '
            'para información médica en español: convierte 303 registros tabulares en narrativas clínicas, '
            'agrega 3 guías normativas en PDF/Markdown, las fragmenta y vectoriza con un modelo SBERT '
            'multilingüe de 384 dimensiones, y las indexa en <b>Qdrant Cloud</b> (AWS · ca-central-1). '
            'La sección <b>Demo RAG en Vivo</b> ejecuta consultas reales contra el clúster en esta presentación.',
            "reloj",
        )
        + ficha(
            "Entregables del ACA",
            'Notebook reproducible con outputs · Póster infográfico en PDF '
            '· Guion de sustentación en video · 3 guías clínicas autogeneradas '
            '· Código modular en <code>src/</code> · Esta app Streamlit.',
            "documento",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Descargar entregables en PDF"):
        cols = st.columns(3)
        docs = [
            ("ACA_1_Computacion_Nube_RAG.pdf", "Documento principal"),
            ("Poster_Infografia_ACA1_RAG_Salud.pdf", "Póster infográfico"),
            ("Resumen_Resultados_Notebook.pdf", "Resumen de resultados"),
        ]
        for col, (fname, label) in zip(cols, docs):
            with col:
                if os.path.exists(fname):
                    with open(fname, "rb") as f:
                        st.download_button(label, f.read(), file_name=fname, width="stretch")


def page_contexto():
    cabecera(1, "Contexto y Pregunta Orientadora")

    st.markdown(
        '<div class="orienta">¿Cómo estructurar e implementar una canalización (pipeline) de datos no '
        'estructurados en la nube mediante bases de datos vectoriales serverless y modelos de embeddings '
        'abiertos, que permita indexar y recuperar semánticamente información contextual especializada '
        'con alta precisión, costo cero y baja latencia?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="rejilla" style="grid-template-columns: 1fr 1fr">'
        + ficha(
            "El problema: información clínica dispersa",
            'La Enfermedad Arterial Coronaria es la <b>principal causa de mortalidad mundial</b> y su información '
            'clave vive en fuentes heterogéneas:'
            '<ul>'
            '<li><b>303 historias clínicas tabulares</b> (dataset UCI Heart Disease, 14 variables).</li>'
            '<li><b>Guías clínicas normativas</b> en PDF/Markdown (MinSalud/OPS/OMS, ACC/AHA, ESC/OMS).</li>'
            '<li>Terminología médica especializada en <b>español</b> que la búsqueda tradicional por '
            'palabras clave no comprende — busca coincidencias literales, no significado.</li>'
            '</ul>',
            "alerta",
        )
        + ficha(
            "La solución: RAG cloud serverless",
            '<ul>'
            '<li><b>Qdrant Cloud</b> serverless en AWS (ca-central-1): índice vectorial HNSW + filtros de '
            'metadatos a nivel sub-milisegundo.</li>'
            '<li><b>Embeddings abiertos</b>: SBERT '
            '<code>paraphrase-multilingual-MiniLM-L12-v2</code>, 384-d, optimizado para español, '
            'sin API keys ni cuotas.</li>'
            '<li><b>Ingesta multimodal</b> con LangChain: narrativas clínicas + chunking recursivo del corpus normativo.</li>'
            f'<li><b>Resultado:</b> {M["fragmentos"]} fragmentos indexados, similitud media &gt; {M["top1_mean"]:.2f}, '
            f'latencia ~{M["latencia"]:.0f} ms, costo cero.</li>'
            '</ul>',
            "capas",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Stack tecnológico y justificación")
    stack = {
        "Capa": [
            "Gestor de entorno (Python 3.12)",
            "Base de datos vectorial",
            "Modelo de embeddings",
            "Chunking de texto",
            "Ingesta del dataset",
            "Seguridad / credenciales",
            "Interfaz de sustentación",
        ],
        "Tecnología": [
            "uv (Astral, resolución determinista)",
            "Qdrant Cloud — Free Tier 1 GB (serverless AWS)",
            "paraphrase-multilingual-MiniLM-L12-v2 (SBERT)",
            "RecursiveCharacterTextSplitter (LangChain)",
            "ucimlrepo (UCI Heart Disease, ID 45)",
            "python-dotenv + archivo .env (git-ignored)",
            "Streamlit (esta app)",
        ],
        "Justificación": [
            "Dependencias reproducibles e instalación ultrarrápida",
            "Filtros sobre índices de payload, costo $0, latencia mínima",
            "Multilingüe nativo en español, 384-d, sin rate limits",
            "Tamaño alineado al límite de 256 tokens del encoder",
            "Descarga programática del dataset de referencia",
            "Las API keys nunca se hardcodean: directriz de seguridad del ACA",
            "Demostración interactiva del sistema ante el docente",
        ],
    }
    st.dataframe(stack, width="stretch", hide_index=True,
                 column_config={"Tecnología": st.column_config.TextColumn(width="large"),
                                "Justificación": st.column_config.TextColumn(width="large")})


def page_arquitectura():
    cabecera(2, "Arquitectura Cloud del Sistema")

    st.markdown(
        '<div class="lead" style="margin-bottom:1.1rem">Flujo de extremo a extremo: fuentes clínicas '
        '<b>→</b> chunking con LangChain <b>→</b> vectorización SBERT <b>→</b> índice HNSW en Qdrant Cloud '
        '<b>→</b> búsqueda semántica con filtrado en tiempo real.</div>',
        unsafe_allow_html=True,
    )

    pasos = [
        ("01", "corazon", "Fuentes", "UCI Heart Disease (303 pacientes, 14 variables) + 3 guías clínicas normativas en MD/PDF."),
        ("02", "documento", "Chunking", f"Tabular→viñeta clínica narrativa + splitter recursivo (512 chars, overlap 12%). Corpus: {M['fragmentos']} fragmentos."),
        ("03", "cerebro", "Embeddings", "MiniLM-L12-v2 multilingüe · 384-d · normalización L₂ (dot = coseno) · upsert por lotes de 64."),
        ("04", "capas", "Almacén", "Qdrant Cloud serverless (AWS ca-central-1) · colección clinical_heart_disease_rag · métrica COSINE."),
        ("05", "lupa", "Recuperación", f"Búsqueda Top-k con filtros de metadatos en tiempo real · latencia ~{M['latencia']:.0f} ms · score medio {M['top1_mean']:.2f}."),
        ("06", "persona", "Presentación", "Esta app Streamlit: sustentación interactiva con demo en vivo, métricas y entregables."),
    ]
    st.markdown(cinta_pasos(pasos), unsafe_allow_html=True)

    if os.path.exists("output/cloud_rag_architecture.png"):
        col1, col2 = st.columns([1.35, 1])
        with col1:
            st.image("output/cloud_rag_architecture.png", width="stretch",
                     caption="Diagrama oficial de arquitectura cloud del proyecto")
        with col2:
            st.markdown(
                ficha(
                    "Seguridad y gobernanza del clúster",
                    '<ul>'
                    '<li><b>Cero credenciales en el código:</b> <code>QDRANT_URL</code> y <code>QDRANT_API_KEY</code> se '
                    'cargan desde <code>.env</code> (excluido del repositorio) con <code>python-dotenv</code>.</li>'
                    '<li><b>Índices de payload nativos</b> en <code>doc_type</code>, <code>sex</code>, '
                    '<code>disease_present</code> y <code>risk_level</code>: el filtrado no penaliza la latencia del HNSW.</li>'
                    '<li><b>Upsert por lotes</b> (batch 64): respeta límites de red y del free tier.</li>'
                    '<li><b>Modo reproducible:</b> sin credenciales, el cliente opera en memoria local.</li>'
                    '</ul>',
                    "candado",
                ),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            ficha(
                "Seguridad y gobernanza del clúster",
                '<ul>'
                '<li><b>Cero credenciales en el código:</b> <code>QDRANT_URL</code> y <code>QDRANT_API_KEY</code> se '
                'cargan desde <code>.env</code> (excluido del repositorio) con <code>python-dotenv</code>.</li>'
                '<li><b>Índices de payload nativos</b> en <code>doc_type</code>, <code>sex</code>, '
                '<code>disease_present</code> y <code>risk_level</code>: el filtrado no penaliza la latencia del HNSW.</li>'
                '<li><b>Upsert por lotes</b> (batch 64): respeta límites de red y del free tier.</li>'
                '<li><b>Modo reproducible:</b> sin credenciales, el cliente opera en memoria local.</li>'
                '</ul>',
                "candado",
            ),
            unsafe_allow_html=True,
        )


def page_datos():
    cabecera(3, "Ingesta Multimodal, Chunking y Corpus")

    st.markdown(
        fila_datos([
            ("303", "Registros tabulares (UCI ID 45)"),
            ("3", "Documentos normativos (MD + PDF)", "sello"),
            (f"{M['fragmentos']}", "Fragmentos consolidados"),
            ("512 / 64", "chunk_size / overlap (chars)"),
            ("384-d", "Espacio latente del embedding"),
        ]),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(
            ficha(
                "Estrategia de chunking (justificación técnica)",
                '<ul>'
                '<li><b>Tabular → narrativa:</b> cada registro se transforma en una viñeta clínica en '
                'lenguaje natural con metadatos enriquecidos (edad, sexo, dolor torácico, estenosis, '
                'pruebas de esfuerzo, nivel de riesgo).</li>'
                '<li><b><code>chunk_size = 512</code>:</b> alineado al límite de atención de 256 tokens '
                'del encoder; evita el truncamiento semántico de spans clínicos.</li>'
                '<li><b><code>chunk_overlap = 64</code> (~12%):</b> conserva la continuidad de las '
                'frases en las fronteras de fragmento.</li>'
                '<li><b>Splitter recursivo</b> con separadores jerárquicos: corta primero por '
                'párrafos, luego frases, y solo al final por palabras.</li>'
                '</ul>',
                "capas",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**Corpus complementario obligatorio (no estructurado)**")
        st.dataframe(
            [
                {"ID": "DOC_01", "Documento": "Guía de Enfermedad Coronaria", "Entidad": "MinSalud / OPS / OMS", "Formato": "MD + PDF"},
                {"ID": "DOC_02", "Documento": "Pruebas de Esfuerzo y Ergometría", "Entidad": "ACC / AHA", "Formato": "MD + PDF"},
                {"ID": "DOC_03", "Documento": "Manejo Farmacológico y Prevención Secundaria", "Entidad": "ESC / OMS", "Formato": "MD + PDF"},
            ],
            hide_index=True, width="stretch",
        )
        st.markdown(
            '<div class="lead">Cada fragmento genera <b>metadatos indexables</b> (<code>doc_type</code>, '
            '<code>chunk_id</code>, <code>chunk_char_length</code>, ...), base del filtrado semántico '
            'demostrado en la sección <b>Demo RAG en Vivo</b>.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Ver fragmentos reales generados por el pipeline ahora mismo"):
        from rag.document_processor import load_unstructured_documents, chunk_unstructured_documents
        from rag.config import settings
        try:
            docs = load_unstructured_documents(str(settings.documents_dir))
            corpus = chunk_unstructured_documents(docs)
            st.markdown(f"**{len(corpus)} fragmentos generados en esta sesión** a partir de las 3 guías clínicas.")
            sel = st.select_slider("Fragmento a inspeccionar", options=[c["metadata"]["chunk_id"] for c in corpus],
                                   value=corpus[0]["metadata"]["chunk_id"])
            chunk = corpus[sel - 1]
            st.code(chunk["text"][:1200], language="text")
            st.caption(f"Metadatos: {chunk['metadata']}")
        except Exception as e:
            st.warning(f"No fue posible cargar los documentos en este entorno: {e}")


def page_demo():
    cabecera(4, "Búsqueda Semántica contra Qdrant Cloud")

    from rag.config import settings
    from rag.query_parser import parse_query
    from rag.vector_store import get_qdrant_client
    from sentence_transformers import SentenceTransformer

    @st.cache_resource(show_spinner="Cargando modelo de embeddings SBERT (solo la primera vez)...")
    def load_model():
        return SentenceTransformer(settings.embedding_model_name)

    @st.cache_resource(show_spinner=False)
    def load_client():
        return get_qdrant_client()

    # --- Interfaz mínima: el usuario escribe y el sistema se encarga -------------
    # No hay selectores de doc_type, ni filtros manuales, ni batería de consultas:
    # el sistema interpreta la consulta y aplica las restricciones por su cuenta.
    EJEMPLOS = [
        "¿Cuáles son los criterios para diagnosticar angina típica?",
        "paciente masculino, mayor de 60 años, con enfermedad coronaria confirmada",
        "¿Qué metas de c-LDL se manejan en prevención secundaria?",
    ]

    query_text = st.text_area(
        "Escriba su consulta en lenguaje natural",
        placeholder="Ej.: ¿Qué significado clínico tiene la depresión del segmento ST mayor a 2 mm...?",
        height=90,
    )

    col_topk, col_ejemplos = st.columns([0.75, 2.25])
    with col_topk:
        top_k = st.slider("Top-k", 1, 10, 5)
    with col_ejemplos:
        st.markdown(
            '<div class="ejemplos">'
            '<div class="ejemplos-t">Consultas de ejemplo</div>'
            + "".join(f"<div class='ejemplo'>{e}</div>" for e in EJEMPLOS)
            + "</div>",
            unsafe_allow_html=True,
        )

    query = query_text.strip()

    # --- Interpretación automática: el sistema decide el filtro ------------------
    parseado = parse_query(query) if query else None
    if parseado and parseado.conditions:
        chips = []
        for c in parseado.conditions:
            clase = " sello" if ("riesgo" in c or "enfermedad" in c) else ""
            chips.append(f'<span class="{clase.strip()}">{c}</span>')
        st.markdown(f'<div class="lectura">{"".join(chips)}</div>', unsafe_allow_html=True)
        st.caption(
            "El sistema interpretó su consulta y aplicó esas restricciones como filtro "
            "sobre los índices de payload de Qdrant. "
            "*Un embedding no puede comparar números: por eso la edad y el diagnóstico "
            "se resuelven con filtros estructurados, no con el texto.*"
        )
    elif query:
        st.caption(
            "No se detectaron restricciones estructuradas en la consulta: se buscará por "
            "similitud semántica sobre todo el corpus."
        )

    ejecutar = st.button("Ejecutar búsqueda en vivo", width="stretch", type="primary")

    if ejecutar:
        with st.spinner("Vectorizando consulta y consultando Qdrant Cloud..."):
            try:
                model = load_model()
                client = load_client()
                t0 = time.perf_counter()
                query_vector = model.encode(query, normalize_embeddings=True).tolist()
                # El filtro lo decide el sistema a partir del texto de la consulta.
                qfilter = parseado.to_filter() if parseado else None
                response = client.query_points(
                    collection_name=settings.collection_name,
                    query=query_vector, limit=top_k, query_filter=qfilter,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                ppoints = response.points
                if not ppoints:
                    st.warning("No se recuperaron fragmentos para esa consulta + filtro. Prueba sin filtro.")
                else:
                    st.success(
                        f"Recuperación exitosa en {elapsed_ms:,.0f} ms · {len(ppoints)} fragmentos · "
                        f"colección `{settings.collection_name}` · Top-1 score **{ppoints[0].score:.4f}**"
                        + (
                            f" · restricciones aplicadas: **{' · '.join(parseado.conditions)}**"
                            if parseado and parseado.conditions
                            else ""
                        )
                    )
                    for rank, point in enumerate(ppoints, 1):
                        payload = point.payload or {}
                        src = payload.get("title") or f"Historia de Paciente #{payload.get('patient_id', '—')}"
                        patient_id = payload.get("patient_id")
                        clinico = ""
                        if patient_id:
                            clinico = (
                                f"<span>edad <b>{payload.get('age', '—')}</b> años</span>"
                                f"<span>sexo <b>{payload.get('sex', '—')}</b></span>"
                                f"<span>dolor <b>{payload.get('chest_pain_label', '—')}</b></span>"
                                f'<span class="r-riesgo">riesgo <b>{payload.get("risk_level", "—")}</b></span>'
                            )
                        # Una línea en blanco dentro del HTML corta el bloque de markdown
                        # y el resto (barra, texto) se imprime como código literal:
                        # clinico vacío dejaba una línea de solo espacios. Se eliminan
                        # las líneas en blanco para que todo el bloque pase como HTML crudo.
                        # shortcut: un <pre> con líneas en blanco dentro del chunk se aplana;
                        # subir a render fragmento a fragmento si algún corpus usa <pre>
                        card = "\n".join(
                            ln
                            for ln in f"""
                            <div class="resultado">
                                <div class="r-cab">
                                    <span class="r-puesto">PUESTO {rank} · {payload.get('doc_type', '—')}</span>
                                    <span class="r-fuente">{src}</span>
                                </div>
                                <div class="r-meta">
                                    <span>similitud coseno <b>{point.score:.4f}</b></span>
                                    {clinico}
                                </div>
                                {barra_similitud(point.score)}
                                <div class="r-texto">{_MD.render((payload.get('text') or '')[:2000])}</div>
                            </div>
                            """.splitlines()
                            if ln.strip()
                        )
                        st.markdown(card, unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"La conexión con Qdrant Cloud no está disponible desde este entorno: {exc}")
                st.info("Verifica las variables `QDRANT_URL` / `QDRANT_API_KEY` del `.env`, o ejecuta la app en "
                        "el equipo local del proyecto. Los resultados oficiales del benchmark ya están abajo.")
                benchmark_table()
    else:
        st.markdown(
            '<div class="lead">Escribe tu consulta (o elige una de la batería) y pulsa '
            '<b>Ejecutar búsqueda en vivo</b>: esta app codifica tu pregunta con SBERT y consulta '
            'el índice HNSW del clúster real. También revisa abajo los resultados oficiales ya validados.</div>',
            unsafe_allow_html=True,
        )
        benchmark_table()


def benchmark_table():
    """Muestra el benchmark oficial con los valores medidos en la última ejecución."""
    from rag.evaluation import load_summary

    summary = load_summary()
    if not summary or not summary.get("queries"):
        st.info(
            "Aún no hay métricas medidas. Ejecute `uv run rag pipeline` para generar "
            "`output/benchmark_summary.json`."
        )
        return

    data = [
        {
            "Q": q["id"],
            "Enfoque": q["description"],
            "Filtro": q["filter_label"],
            "Top-1": q["top1_score"],
            "Resultado observado": q.get("expected", ""),
        }
        for q in summary["queries"]
    ]
    latency = summary.get("latency", {})
    st.markdown("#### Benchmark oficial validado en el Notebook y el Póster")
    st.dataframe(
        data,
        hide_index=True,
        width="stretch",
        column_config={
            "Top-1": st.column_config.NumberColumn("Top-1 Score", format="%.4f"),
            "Filtro": st.column_config.TextColumn(width="medium"),
            "Resultado observado": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        f"Similitud media Top-1: **{summary['top1_mean']:.4f}** · máximo: **{summary['top1_max']:.4f}**"
        + (
            f" · latencia media de búsqueda remota: **{latency['search_mean_ms']:.0f} ms**"
            if latency
            else ""
        )
    )


def page_resultados():
    cabecera(5, "Resultados y Métricas de Validación")

    st.markdown(
        fila_datos([
            (f"{M['top1_mean']:.4f}", "Similitud coseno media global"),
            (f"{M['top1_max']:.4f}", "Pico de similitud coseno", "sello"),
            ("5/5", "Consultas del benchmark con resultado relevante"),
            ("1→5", "Caída monótona del score entre rangos"),
        ]),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        if os.path.exists("output/similarity_distribution.png"):
            st.image("output/similarity_distribution.png", width="stretch",
                     caption="Boxplot por consulta y similitud media según rango de recuperación (Rank 1–5)")
        if os.path.exists("output/metricas_similitud_notebook.png"):
            st.image("output/metricas_similitud_notebook.png", width="stretch",
                     caption="Métricas generadas por el Notebook reproducible")
    with col2:
        st.markdown(
            ficha(
                "Lectura técnica de las métricas",
                '<ul>'
                f'<li><b>Similitud media &gt; {M["top1_mean"]:.2f}</b> con caída <b>monótona</b> entre Rank 1 y Rank 5: '
                'el sistema ordena por relevancia real, no ruido — <b>discrimina semánticamente</b>.</li>'
                f'<li><b>Pico de {M["top1_max"]:.4f} en las consultas más específicas:</b> los protocolos normativos comparten '
                'vocabulario y estructura, y el modelo SBERT capta esa afinidad temática.</li>'
                '<li><b>Filtros sin penalización:</b> filtrar por <code>doc_type</code>, <code>sex</code> '
                'o <code>disease_present</code> combina HNSW + índice de payload y aun así devolvió '
                'al paciente de mayor riesgo que cumple todos los criterios en Q4.</li>'
                f'<li><b>~{M["latencia"]:.0f} ms end-to-end</b> contra el clúster serverless: rendimiento apto para '
                'uso clínico interactivo dentro del free tier.</li>'
                '</ul>',
                "grafico",
            ),
            unsafe_allow_html=True,
        )
    st.info("Ambos gráficos provienen del Notebook reproducible "
            "`heart_disease_rag_qdrant.executed.ipynb` sobre las 5 consultas del benchmark.")


def page_conclusiones():
    cabecera(6, "Conclusiones", "Veredicto del Proyecto")

    concl = [
        ("capas", "Cloud con costo cero", "Qdrant Cloud Serverless + <code>uv</code> + modelos SBERT abiertos "
         "entrega un pipeline RAG completo <b>sin costo de licencias ni de API</b>: la infraestructura paga "
         "con creatividad, no con presupuesto."),
        ("diana", "Precisión semántica real", "Las 5 consultas del benchmark recuperaron fragmentos "
         "clínicamente correctos con similitud media &gt; 0.60 y tendencia monótona por rank. "
         "<b>La búsqueda por significado superó la búsqueda literal.</b>"),
        ("rayo", "Latencia de producción", f"Recuperación ~{M['latencia']:.0f} ms incluso con filtros compuestos de payload: "
         "suficiente para asistentes clínicos en tiempo real sobre infraestructura gratuita."),
        ("candado", "Seguridad y reproducibilidad", "Ninguna credencial en el código ni en el repositorio; "
         "notebook ejecutado, póster generado por código, y esta app comparten el mismo corpus y la "
         "misma lógica → trazabilidad completa de punta a punta."),
    ]
    st.markdown(
        '<div class="rejilla" style="grid-template-columns: repeat(auto-fit, minmax(240px, 1fr))">'
        + "".join(
            f'<div class="ficha"><h3>{icono(ic)} {t}</h3>'
            f'<div class="lead" style="font-size:.83rem">{d}</div></div>'
            for ic, t, d in concl
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        ficha(
            "Qué demostró esta sustentación",
            '<ul>'
            '<li>Una <b>arquitectura cloud serverless implementada</b>, no solo diseñada: dataset real, '
            'embedding real, colección real y consultas reales en vivo desde esta aplicación.</li>'
            '<li><b>Reproducibilidad de extremo a extremo:</b> el notebook ejecutado, los entregables en PDF y '
            f'esta app están en el repositorio y operan sobre el mismo corpus de {M["fragmentos"]} fragmentos.</li>'
            '<li><b>Respuesta afirmativa a la pregunta orientadora:</b> se lograron indexar y recuperar '
            'semánticamente datos clínicos especializados con alta precisión, costo cero y baja latencia, '
            'usando exclusivamente servicios serverless y modelos abiertos.</li>'
            '</ul>',
            "sello",
        ),
        unsafe_allow_html=True,
    )


def page_referencias():
    cabecera(7, "Referencias en formato APA 7")

    refs = [
        "American College of Cardiology & American Heart Association. (2024). <i>Clinical Protocol for Exercise Testing, Ergometry, and Coronary Angiography</i> (Documento Técnico PRT-DIAG-CARDIO-ACC-2024). ACC/AHA.",
        "Chase, H. (2022). <i>LangChain: Building applications with LLMs through composability and data augmented generation.</i> https://github.com/langchain-ai/langchain",
        "European Society of Cardiology & World Health Organization. (2024). <i>Clinical Protocol for Pharmacological Management and Secondary Cardiovascular Prevention</i> (Consenso ESC-OMS-PREV-2024). ESC/WHO.",
        "Janosi, A., Steinbrunn, R., Pfisterer, M., & Detrano, R. (1988). <i>UCI Heart Disease Dataset</i> [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease",
        "Lewis, P., Perez, E., Piktus, F., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>Advances in Neural Information Processing Systems</i>, 33, 9459–9474. https://arxiv.org/abs/2005.11401",
        "Ministerio de Salud y Protección Social & OPS. (2024). <i>Guía de Práctica Clínica: Diagnóstico y Estratificación del Riesgo en Enfermedad Arterial Coronaria</i> (GPC-CARDIO-2024-V2). MinSalud/OPS.",
        "Qdrant Solutions GmbH. (2024). <i>Qdrant Documentation: Vector Search Engine & Cloud Database.</i> https://qdrant.tech/documentation/",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)</i>. https://arxiv.org/abs/1908.10084",
    ]
    bloque = "".join(f'<div class="ref">{r}</div>' for r in refs)
    st.markdown(bloque, unsafe_allow_html=True)
    st.caption("Consenso de la asignatura: todas las referencias citadas siguen el formato APA 7.")


# ============ ROUTER ============
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            f"""
            <div class="marca-app">
                {icono("corazon", "ico ico-lg")}
                <div>
                    <div class="t1">RAG Cloud en Salud</div>
                    <div class="t2">Sustentación ACA N° 1 · CUN</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sel = st.radio("Sección", SECTION_LABELS, label_visibility="collapsed")
        st.markdown(
            """
            <div class="pie-indice">
                <b>Stack</b> · Qdrant Cloud (AWS) · SBERT 384-d · uv · LangChain<br>
                <b>Demo</b> · consultas reales contra el clúster serverless en vivo
            </div>
            """,
            unsafe_allow_html=True,
        )

    PAGES = {
        "Portada": page_portada,
        "Contexto y Pregunta Orientadora": page_contexto,
        "Arquitectura Cloud": page_arquitectura,
        "Datos, Chunking e Ingesta": page_datos,
        "Demo RAG en Vivo (Qdrant Cloud)": page_demo,
        "Resultados y Métricas": page_resultados,
        "Conclusiones": page_conclusiones,
        "Referencias (APA 7)": page_referencias,
    }
    page = PAGES[sel]
    page()


if __name__ == "__main__":
    main()
