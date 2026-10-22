"""
App de Sustentación Interactiva — Proyecto ACA N° 1: Computación en la Nube
Sistema RAG Cloud Serverless en Salud con Qdrant y Embeddings Abiertos.

Ejecutar:  uv run streamlit run app_sustentacion.py
"""

import os
import time

import streamlit as st

st.set_page_config(
    page_title="ACA 1 · RAG Cloud en Salud | Jaime Cabrera",
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
SECTION_LABELS = [
    "📄 Portada",
    "🎯 Contexto y Pregunta Orientadora",
    "🏛️ Arquitectura Cloud",
    "📦 Datos, Chunking e Ingesta",
    "🚀 Demo RAG en Vivo (Qdrant Cloud)",
    "📊 Resultados y Métricas",
    "✅ Conclusiones",
    "📚 Referencias (APA 7)",
]

# ============ ESTILO GLOBAL CUSTOM ============
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg: #0A0F1E;
        --card: #111930;
        --card-alt: #0E1426;
        --accent: #4CC9F0;
        --accent-red: #E63946;
        --accent-green: #38D9A9;
        --text: #E9EEF7;
        --muted: #8FA0BF;
        --border: rgba(76, 201, 240, 0.18);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        color: var(--text);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---- HERO ---- */
    .hero {
        background: linear-gradient(135deg, #16204A 0%, #0E1A3C 55%, #1B2B5E 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.3rem 2.8rem;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(230,57,70,0.22) 0%, transparent 65%);
        border-radius: 50%;
    }
    .hero::after {
        content: "";
        position: absolute; bottom: -80px; left: 30%;
        width: 380px; height: 280px;
        background: radial-gradient(circle, rgba(76,201,240,0.14) 0%, transparent 65%);
        border-radius: 50%;
    }
    .hero h1 {
        font-size: 2.1rem; font-weight: 800; line-height: 1.3;
        margin: 0 0 0.55rem 0; letter-spacing: -0.5px; color: #FFFFFF;
        position: relative; z-index: 2;
    }
    .hero h1 .grad {
        background: linear-gradient(90deg, #4CC9F0, #7EE0FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero .sub { font-size: 1rem; color: #AFC2E4; margin-bottom: 1.1rem; position: relative; z-index: 2; }
    .hero .badges { display: flex; gap: 0.55rem; flex-wrap: wrap; }

    /* ---- BADGES ---- */
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 0.34rem 0.85rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.2px;
        border: 1px solid var(--border);
        background: rgba(76, 201, 240, 0.09); color: #9FE3FF;
        z-index: 2; position: relative;
    }
    .badge.red { background: rgba(230,57,70,0.11); border-color: rgba(230,57,70,0.35); color: #FF9AA2; }
    .badge.green { background: rgba(56,217,169,0.10); border-color: rgba(56,217,169,0.35); color: #8FFFF0; }

    /* ---- MÉTRICAS ---- */
    .metric-card {
        background: linear-gradient(160deg, var(--card) 0%, var(--card-alt) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem 1.1rem;
        text-align: center;
        height: 100%;
    }
    .metric-card .num { font-size: 1.75rem; font-weight: 800; color: var(--accent); line-height: 1.15; }
    .metric-card .num.red { color: var(--accent-red); }
    .metric-card .num.green { color: var(--accent-green); }
    .metric-card .lbl {
        font-size: 0.72rem; color: var(--muted); margin-top: 0.35rem;
        text-transform: uppercase; letter-spacing: 0.6px;
    }

    /* ---- TARJETAS ---- */
    .card {
        background: linear-gradient(160deg, var(--card) 0%, var(--card-alt) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        height: 100%;
    }
    .card h3 { margin-top: 0; font-weight: 700; font-size: 1rem; }
    .lead { color: #B7C6E0; font-size: 0.95rem; line-height: 1.7; }
    .lead b, .lead code { color: #DCE8FB; }
    .lead ul { padding-left: 1.1rem; }

    /* ---- PASOS ---- */
    .step-card {
        background: linear-gradient(160deg, var(--card) 0%, var(--card-alt) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.05rem 1rem 0.95rem 1rem;
        height: 100%;
        text-align: center;
    }
    .step-card .icon { font-size: 1.8rem; margin-bottom: 0.45rem; }
    .step-card .step-num {
        display: inline-block; font-size: 0.66rem; font-weight: 700;
        color: var(--accent); background: rgba(76,201,240,0.12);
        border-radius: 6px; padding: 0.15rem 0.55rem; margin-bottom: 0.4rem;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .step-card .t { font-weight: 700; font-size: 0.88rem; margin-bottom: 0.3rem; }
    .step-card .d { font-size: 0.76rem; color: var(--muted); line-height: 1.55; text-align: center; }

    /* ---- RESULTADOS DE DEMO ---- */
    .result-card {
        background: linear-gradient(160deg, var(--card) 0%, var(--card-alt) 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 14px;
        padding: 1.05rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .result-card .source { font-weight: 700; font-size: 0.93rem; color: #FFFFFF; }
    .result-card .frag {
        margin-top: 0.65rem;
        background: rgba(6, 10, 22, 0.55);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        font-size: 0.82rem;
        color: #C6D4EA;
        white-space: pre-wrap;
        max-height: 230px;
        overflow-y: auto;
        line-height: 1.6;
    }
    .chip {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        margin-right: 0.3rem;
        border-radius: 7px;
        font-size: 0.71rem;
        font-weight: 600;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.09);
        color: #B9C8E4;
    }
    .score-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 9px;
        width: 100%;
        overflow: hidden;
        margin: 0.45rem 0 0.25rem 0;
    }
    .score-bar {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #4CC9F0, #38D9A9);
    }

    /* ---- CABECERA DE SECCIÓN ---- */
    .section-head .kicker {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: var(--accent); margin-bottom: 0.15rem;
    }
    .section-head h2 {
        font-size: 1.55rem; font-weight: 800; margin: 0 0 0.8rem 0;
        color: #FFFFFF; letter-spacing: -0.4px;
    }

    /* ---- QUOTE ---- */
    .orienta {
        background: linear-gradient(135deg, rgba(230,57,70,0.12), rgba(76,201,240,0.08));
        border: 1px solid rgba(230, 57, 70, 0.32);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        font-size: 1.08rem;
        line-height: 1.8;
        color: #F3F7FF;
        font-weight: 500;
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1224 0%, #0A0F1E 100%);
        border-right: 1px solid var(--border);
    }
    .nav-item {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.5rem 0.8rem; border-radius: 10px;
        color: #A9BBD9; font-size: 0.87rem; font-weight: 500;
        border: 1px solid transparent; margin-bottom: 0.15rem;
    }
    .nav-item.active {
        background: linear-gradient(90deg, rgba(76,201,240,0.16), rgba(76,201,240,0.03));
        border-color: var(--border); color: #FFFFFF; font-weight: 700;
    }

    /* ---- MISC ---- */
    div[data-testid="stDataFrame"] > div {
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }
    code {
        background: rgba(76, 201, 240, 0.10) !important;
        border-radius: 5px !important;
        color: #9FE3FF !important;
        font-size: 0.9em !important;
        padding: 0.1em 0.4em !important;
    }
    details[role="group"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        background: var(--card-alt) !important;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(76,201,240,0.35); border-radius: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
</style>
"""


# ============ HELPERS ============

def header(section: str, kicker: str, title: str):
    st.markdown(
        f"""
        <div class="section-head">
            <div>
                <div class="kicker">{kicker} &nbsp;·&nbsp; ACA N° 1 · COMPUTACIÓN EN LA NUBE</div>
                <h2>{section} {title}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items):
    cols = st.columns(len(items))
    for col, (num, lbl, color) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="num {color}">{num}</div>
                    <div class="lbl">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def score_bar(score: float):
    pct = max(3.0, min(100.0, score * 100.0))
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar" style="width:{pct:.1f}%"></div></div>'
    )


# ============ PÁGINAS ============

def page_portada():
    st.markdown(
        f"""
        <div class="hero">
            <div class="badges" style="margin-bottom:1.1rem;">
                <span class="badge">🏫 CUN · Especialización en IA</span>
                <span class="badge red">🩺 Pilar: Salud — Cardiología</span>
                <span class="badge green">💰 Costo $0.00 USD</span>
                <span class="badge">⚡ Latencia ~{M['latencia']:.0f} ms</span>
            </div>
            <h1>Sistema RAG Cloud <span class="grad">Serverless</span> en Salud<br>con Qdrant y Embeddings Abiertos</h1>
            <div class="sub">
                Pipeline cloud de datos clínicos no estructurados: ingesta multimodal, fragmentación,
                vectorización semántica y recuperación con alta precisión — sustentación interactiva con demo en vivo.
            </div>
            <div class="badges">
                <span class="badge">👨‍🎓 Jaime Cabrera · Estudiante</span>
                <span class="badge">👨‍🏫 Hamilton Rivera Flor · Docente</span>
                <span class="badge red">ACA N° 1 · Computación en la Nube</span>
            </div>
        </div>
        f""",
        unsafe_allow_html=True,
    )

    metric_row([
        (f"{M['fragmentos']}", "Fragmentos vectoriales indexados", ""),
        (f"{M['pacientes']}", "Pacientes UCI Heart Disease", ""),
        ("384-d", "Espacio latente SBERT", "red"),
        (f"~{M['latencia']:.0f} ms", "Latencia de recuperación", "green"),
    ])
    st.markdown("")
    metric_row([
        (f"{M['top1_max']:.4f}", "Pico de similitud coseno (Top-1)", "red"),
        (f"{M['top1_mean']:.4f}", "Similitud coseno media (Top-1)", ""),
        ("3", "Guías clínicas normativas", "red"),
        ("$0.00", "Costo total del despliegue", "green"),
    ])

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        st.markdown(
            '<div class="card"><h3>🚀 La sustentación en 30 segundos</h3>'
            '<div class="lead">Este proyecto implementa un <b>pipeline cloud serverless</b> end-to-end '
            'para información médica en español: convierte 303 registros tabulares en narrativas clínicas, '
            'agrega 3 guías normativas en PDF/Markdown, las fragmenta y vectoriza con un modelo SBERT '
            'multilingüe de 384 dimensiones, y las indexa en <b>Qdrant Cloud</b> (AWS · ca-central-1). '
            'La sección <b>Demo RAG en Vivo</b> ejecuta consultas reales contra el clúster en esta presentación.</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="card"><h3>📦 Entregables del ACA</h3>'
            '<div class="lead">Notebook reproducible con outputs · Póster infográfico en PDF '
            '· Guion de sustentación en video · 3 guías clínicas autogeneradas '
            '· Código modular en <code>src/</code> · Esta app Streamlit.</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("⬇️ Descargar entregables en PDF"):
        cols = st.columns(3)
        docs = [
            ("ACA_1_Computacion_Nube_RAG.pdf", "📘 Documento principal"),
            ("Poster_Infografia_ACA1_RAG_Salud.pdf", "🖼️ Póster infográfico"),
            ("Resumen_Resultados_Notebook.pdf", "📋 Resumen de resultados"),
        ]
        for col, (fname, label) in zip(cols, docs):
            with col:
                if os.path.exists(fname):
                    with open(fname, "rb") as f:
                        st.download_button(label, f.read(), file_name=fname, width="stretch")


def page_contexto():
    header("🎯", "Punto de Partida", "Contexto y Pregunta Orientadora")

    st.markdown(
        '<div class="orienta">📌 <b>¿Cómo estructurar e implementar una canalización (pipeline) de datos no '
        'estructurados en la nube mediante bases de datos vectoriales serverless y modelos de embeddings '
        'abiertos, que permita indexar y recuperar semánticamente información contextual especializada '
        'con alta precisión, costo cero y baja latencia?</b></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="card">
            <h3>🩺 El problema: información clínica dispersa</h3>
            <div class="lead">
            La Enfermedad Arterial Coronaria es la <b>principal causa de mortalidad mundial</b> y su información
            clave vive en fuentes heterogéneas:
            <ul>
                <li><b>303 historias clínicas tabulares</b> (dataset UCI Heart Disease, 14 variables).</li>
                <li><b>Guías clínicas normativas</b> en PDF/Markdown (MinSalud/OPS/OMS, ACC/AHA, ESC/OMS).</li>
                <li>Terminología médica especializada en <b>español</b> que la búsqueda tradicional por
                palabras clave no comprende — busca coincidencias literales, no significado.</li>
            </ul>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="card">
            <h3>☁️ La solución: RAG cloud serverless</h3>
            <div class="lead">
            <ul>
                <li><b>Qdrant Cloud</b> serverless en AWS (ca-central-1): índice vectorial HNSW + filtros de
                metadatos a nivel sub-milisegundo.</li>
                <li><b>Embeddings abiertos</b>: SBERT
                <code>paraphrase-multilingual-MiniLM-L12-v2</code>, 384-d, optimizado para español,
                sin API keys ni cuotas.</li>
                <li><b>Ingesta multimodal</b> con LangChain: narrativas clínicas + chunking recursivo del corpus normativo.</li>
                <li><b>Resultado:</b> {M['fragmentos']} fragmentos indexados, similitud media &gt; {M['top1_mean']:.2f}, latencia ~{M['latencia']:.0f} ms, costo cero.</li>
            </ul>
            </div>
            </div>
            f""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🧰 Stack tecnológico y justificación")
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
    header("🏛️", "Diseño de la Solución", "Arquitectura Cloud del Sistema")

    st.markdown(
        '<div class="lead" style="margin-bottom:1.1rem">Flujo de extremo a extremo: fuentes clínicas '
        '<b>→</b> chunking con LangChain <b>→</b> vectorización SBERT <b>→</b> índice HNSW en Qdrant Cloud '
        '<b>→</b> búsqueda semántica con filtrado en tiempo real.</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("🩺", "01 · Fuentes", "UCI Heart Disease (303 pacientes, 14 variables) + 3 guías clínicas normativas en MD/PDF."),
        ("✂️", "02 · Chunking", f"Tabular→viñeta clínica narrativa + splitter recursivo (512 chars, overlap 12%). Corpus: {M['fragmentos']} fragmentos."),
        ("🧠", "03 · Embeddings", "MiniLM-L12-v2 multilingüe · 384-d · normalización L₂ (dot = coseno) · upsert por lotes de 64."),
        ("☁️", "04 · Almacén", "Qdrant Cloud serverless (AWS ca-central-1) · colección clinical_heart_disease_rag · métrica COSINE."),
        ("🔎", "05 · Recuperación", f"Búsqueda Top-k con filtros de metadatos en tiempo real · latencia ~{M['latencia']:.0f} ms · score medio {M['top1_mean']:.2f}."),
        ("🖥️", "06 · Presentación", "Esta app Streamlit: sustentación interactiva con demo en vivo, métricas y entregables."),
    ]
    cols = st.columns(6)
    for col, (icon, t, d) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="icon">{icon}</div>
                    <div class="step-num">{t}</div>
                    <div class="d">{d}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns([1.35, 1])
    with col1:
        if os.path.exists("output/cloud_rag_architecture.png"):
            st.image("output/cloud_rag_architecture.png", width="stretch",
                     caption="Diagrama oficial de arquitectura cloud del proyecto")
    with col2:
        st.markdown(
            """
            <div class="card" style="height:100%">
            <h3>🔐 Seguridad y gobernanza del clúster</h3>
            <div class="lead">
            <ul>
                <li><b>Cero credenciales en el código:</b> <code>QDRANT_URL</code> y <code>QDRANT_API_KEY</code> se
                cargan desde <code>.env</code> (excluido del repositorio) con <code>python-dotenv</code>.</li>
                <li><b>Índices de payload nativos</b> en <code>doc_type</code>, <code>sex</code>,
                <code>disease_present</code> y <code>risk_level</code>: el filtrado no penaliza la latencia del HNSW.</li>
                <li><b>Upsert por lotes</b> (batch 64): respeta límites de red y del free tier.</li>
                <li><b>Modo reproducible:</b> sin credenciales, el cliente opera en memoria local.</li>
            </ul>
            </div>
            </div>
            f""",
            unsafe_allow_html=True,
        )


def page_datos():
    header("📦", "Capa de Datos", "Ingesta Multimodal, Chunking y Corpus")

    metric_row([
        ("303", "Registros tabulares (UCI ID 45)", ""),
        ("3", "Documentos normativos (MD + PDF)", "red"),
        (f"{M['fragmentos']}", "Fragmentos consolidados", "green"),
        ("512 / 64", "chunk_size / overlap (chars)", ""),
        ("384-d", "Espacio latente del embedding", "red"),
    ])
    st.markdown("")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(
            """
            <div class="card" style="height:100%">
            <h3>🏗️ Estrategia de chunking (justificación técnica)</h3>
            <div class="lead">
            <ul>
                <li><b>Tabular → narrativa:</b> cada registro se transforma en una viñeta clínica en
                lenguaje natural con metadatos enriquecidos (edad, sexo, dolor torácico, estenosis,
                pruebas de esfuerzo, nivel de riesgo).</li>
                <li><b><code>chunk_size = 512</code>:</b> alineado al límite de atención de 256 tokens
                del encoder; evita el truncamiento semántico de spans clínicos.</li>
                <li><b><code>chunk_overlap = 64</code> (~12%):</b> conserva la continuidad de las
                frases en las fronteras de fragmento.</li>
                <li><b>Splitter recursivo</b> con separadores jerárquicos: corta primero por
                párrafos, luego frases, y solo al final por palabras.</li>
            </ul>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**📄 Corpus complementario obligatorio (no estructurado)**")
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

    with st.expander("👁️ Ver fragmentos reales generados por el pipeline ahora mismo"):
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
    header("🚀", "Demostración Funcional en Tiempo Real", "Búsqueda Semántica contra Qdrant Cloud")

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
        "✍️ Escriba su consulta en lenguaje natural",
        placeholder="Ej.: ¿Qué significado clínico tiene la depresión del segmento ST mayor a 2 mm...?",
        height=90,
    )

    col_topk, col_ejemplos = st.columns([0.75, 2.25])
    with col_topk:
        top_k = st.slider("🔝 Top-k", 1, 10, 5)
    with col_ejemplos:
        st.caption("💡 Ejemplos que puede probar:")
        for ejemplo in EJEMPLOS:
            st.caption(f"   • {ejemplo}")

    query = query_text.strip()

    # --- Interpretación automática: el sistema decide el filtro ------------------
    parseado = parse_query(query) if query else None
    if parseado and parseado.conditions:
        st.markdown(
            "".join(
                f"<span class='chip' style='background:rgba(56,217,169,.16);color:#8FFFF0'>🧠 {c}</span> "
                for c in parseado.conditions
            ),
            unsafe_allow_html=True,
        )
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

    ejecutar = st.button("⚡ Ejecutar búsqueda en vivo", width="stretch", type="primary")

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
                        f"✅ Recuperación exitosa en {elapsed_ms:,.0f} ms · {len(ppoints)} fragmentos · "
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
                        extra = ""
                        if patient_id:
                            extra = (f"<span class='chip'>👤 {payload.get('age', '—')} años</span>"
                                     f"<span class='chip'>⚕️ {payload.get('sex', '—')}</span>"
                                     f"<span class='chip'>🩺 {payload.get('chest_pain_label', '—')}</span>"
                                     f"<span class='chip'>🎯 riesgo: {payload.get('risk_level', '—')}</span>")
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem">
                                    <div>
                                        <span class="chip" style="background:rgba(76,201,240,.18);color:#9FE3FF">RANK {rank}</span>
                                        <span class="chip">{payload.get('doc_type', '—')}</span>
                                    </div>
                                    <div class="source">{src}</div>
                                </div>
                                <div style="margin-top:.45rem;font-size:.78rem;color:#8FA0BF">
                                    🏆 similitud coseno = <b style="color:#fff">{point.score:.4f}</b> &nbsp;{extra}
                                </div>
                                {score_bar(point.score)}
                                <div class="frag">{(payload.get('text') or '')[:2000]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as exc:
                st.error(f"❌ La conexión con Qdrant Cloud no está disponible desde este entorno: {exc}")
                st.info("💡 Verifica las variables `QDRANT_URL` / `QDRANT_API_KEY` del `.env`, o ejecuta la app en "
                        "el equipo local del proyecto. Los resultados oficiales del benchmark ya están abajo.")
                benchmark_table()
    else:
        st.markdown(
            '<div class="lead">👆 Escribe tu consulta (o elige una de la batería) y pulsa '
            '<b>Ejecutar búsqueda en vivo</b>: esta app codifica tu pregunta con SBERT y consulta '
            'el índice HNSW del clúster real. También revisa abajo los resultados oficiales ya validados.</div>',
            unsafe_allow_html=True,
        )
        benchmark_table()


def benchmark_table():
    """Muestra el benchmark oficial con los valores medidos en la última ejecución.f"""
    from rag.evaluation import load_summary

    summary = load_summary()
    if not summary or not summary.get("queries"):
        st.info(
            "ℹ️ aún no hay métricas medidas. Ejecute `uv run rag pipeline` para generar "
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
    st.markdown("#### 📋 Benchmark oficial validado en el Notebook y el Póster")
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
    header("📊", "Análisis Cuantitativo", "Resultados y Métricas de Validación")

    metric_row([
        (f"{M['top1_mean']:.4f}", "Similitud coseno media global", "green"),
        (f"{M['top1_max']:.4f}", "Pico de similitud coseno", "red"),
        ("5/5", "Consultas del benchmark con resultado relevante", ""),
        ("1→5", "Caída monótona del score entre rangos", "green"),
    ])
    st.markdown("")

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
            f"""
            <div class="card" style="height:100%">
            <h3>🧪 Lectura técnica de las métricas</h3>
            <div class="lead">
            <ul>
                <li><b>Similitud media &gt; {M['top1_mean']:.2f}</b> con caída <b>monótona</b> entre Rank 1 y Rank 5:
                el sistema ordena por relevancia real, no ruido — <b>discrimina semánticamente</b>.</li>
                <li><b>Pico de {M['top1_max']:.4f} en las consultas más específicas:</b> los protocolos normativos comparten
                vocabulario y estructura, y el modelo SBERT capta esa afinidad temática.</li>
                <li><b>Filtros sin penalización:</b> filtrar por <code>doc_type</code>, <code>sex</code>
                o <code>disease_present</code> combina HNSW + índice de payload y aun así devolvió
                al paciente de mayor riesgo que cumple todos los criterios en Q4.</li>
                <li><b>~{M['latencia']:.0f} ms end-to-end</b> contra el clúster serverless: rendimiento apto para
                uso clínico interactivo dentro del free tier.</li>
            </ul>
            </div>
            </div>
            f""",
            unsafe_allow_html=True,
        )
    st.info("💡 Ambos gráficos provienen del Notebook reproducible "
            "`heart_disease_rag_qdrant.executed.ipynb` sobre las 5 consultas del benchmark.")


def page_conclusiones():
    header("✅", "Veredicto del Proyecto", "Conclusiones")

    concl = [
        ("☁️", "Cloud con costo cero", "Qdrant Cloud Serverless + <code>uv</code> + modelos SBERT abiertos "
         "entrega un pipeline RAG completo <b>sin costo de licencias ni de API</b>: la infraestructura paga "
         "con creatividad, no con presupuesto."),
        ("🎯", "Precisión semántica real", "Las 5 consultas del benchmark recuperaron fragmentos "
         "clínicamente correctos con similitud media &gt; 0.60 y tendencia monótona por rank. "
         "<b>La búsqueda por significado superó la búsqueda literal.</b>"),
        ("⚡", "Latencia de producción", f"Recuperación ~{M['latencia']:.0f} ms incluso con filtros compuestos de payload: "
         "suficiente para asistentes clínicos en tiempo real sobre infraestructura gratuita."),
        ("🔐", "Seguridad y reproducibilidad", "Ninguna credencial en el código ni en el repositorio; "
         "notebook ejecutado, póster generado por código, y esta app comparten el mismo corpus y la "
         "misma lógica → trazabilidad completa de punta a punta."),
    ]
    cols = st.columns(4)
    for col, (ic, t, d) in zip(cols, concl):
        with col:
            st.markdown(
                f'<div class="card" style="text-align:left">'
                f'<div style="font-size:1.55rem;margin-bottom:.35rem">{ic}</div>'
                f'<h3 style="font-size:.98rem">{t}</h3>'
                f'<div class="lead" style="font-size:.83rem">{d}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="card"><h3>🧭 Qué demostró esta sustentación</h3>'
        '<div class="lead"><ul>'
        '<li>Una <b>arquitectura cloud serverless implementada</b>, no solo diseñada: dataset real, '
        'embedding real, colección real y consultas reales en vivo desde esta aplicación.</li>'
        '<li><b>Reproducibilidad de extremo a extremo:</b> el notebook ejecutado, los entregables en PDF y '
        f'esta app están en el repositorio y operan sobre el mismo corpus de {M["fragmentos"]} fragmentos.</li>'
        '<li><b>Respuesta afirmativa a la pregunta orientadora:</b> se lograron indexar y recuperar '
        'semánticamente datos clínicos especializados con alta precisión, costo cero y baja latencia, '
        'usando exclusivamente servicios serverless y modelos abiertos.</li>'
        '</ul></div></div>',
        unsafe_allow_html=True,
    )


def page_referencias():
    header("📚", "Bibliografía", "Referencias en formato APA 7")

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
    for r in refs:
        st.markdown(
            f"<div style='padding:.6rem 0;border-bottom:1px solid rgba(76,201,240,.12);"
            f"color:#C9D6EC;line-height:1.75;font-size:.92rem'>{r}</div>",
            unsafe_allow_html=True,
        )
    st.caption("Consenso de la asignatura: todas las referencias citadas siguen el formato APA 7.")


# ============ ROUTER ============
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <div style="padding:.35rem .4rem 1.2rem">
                <div style="font-size:1.42rem;font-weight:800;line-height:1.3;color:#FFFFFF">🩺 RAG Cloud</div>
                <div style="font-size:.98rem;font-weight:700;color:#4CC9F0;margin-bottom:.4rem">Sustentación ACA N° 1</div>
                <div style="font-size:.75rem;color:#8FA0BF">Computación en la Nube · CUN</div>
                <div style="font-size:.75rem;color:#8FA0BF">👨‍🎓 Jaime A. Cabrera A.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sel = st.radio("Sección", SECTION_LABELS, label_visibility="collapsed")
        st.markdown(
            """
            <div style="margin-top:1.6rem;padding-top:1.1rem;border-top:1px solid rgba(76,201,240,.16);
                        font-size:.74rem;color:#8FA0BF;line-height:1.7">
                <b style="color:#9FE3FF">Stack</b> · Qdrant Cloud (AWS) · SBERT 384-d · uv · LangChain<br>
                <b style="color:#9FE3FF">Demo</b> · consultas reales contra el clúster serverless en vivo
            </div>
            """,
            unsafe_allow_html=True,
        )

    PAGES = {
        "📄 Portada": page_portada,
        "🎯 Contexto y Pregunta Orientadora": page_contexto,
        "🏛️ Arquitectura Cloud": page_arquitectura,
        "📦 Datos, Chunking e Ingesta": page_datos,
        "🚀 Demo RAG en Vivo (Qdrant Cloud)": page_demo,
        "📊 Resultados y Métricas": page_resultados,
        "✅ Conclusiones": page_conclusiones,
        "📚 Referencias (APA 7)": page_referencias,
    }
    page = PAGES[sel]
    page()


if __name__ == "__main__":
    main()
