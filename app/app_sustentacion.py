"""Agente RAG clínico — demo en vivo contra Qdrant Cloud.

Interfaz única de la app: un chat. El agente interpreta la consulta, aplica
los filtros de metadatos que detecta, recupera los fragmentos reales del
clúster y el modelo generativo remoto redacta una única respuesta natural
y estructurada. Los fragmentos crudos quedan colapsados como fuentes
verificables. El diseño vive en `app/theme.py`.
"""

import streamlit as st
from markdown_it import MarkdownIt  # dependencia garantizada por streamlit

from rag.config import settings
from rag.generator import generar_respuesta
from rag.query_parser import parse_query
from rag.vector_store import get_qdrant_client, semantic_search
from theme import CUSTOM_CSS, icono

# El fragmento llega con markdown (###, listas) desde Qdrant: renderizado a HTML
# aquí y con html=False, para que Streamlit vea un bloque HTML continuo (si no,
# el markdown crudo dentro del <div> se imprime como texto literal).
_MD = MarkdownIt("commonmark", {"html": False})

#: Fragmentos que devuelve el agente por consulta.
TOP_K = 5

EJEMPLOS = [
    ("Angina típica", "¿Cuáles son los criterios para diagnosticar angina típica?"),
    ("Paciente de riesgo", "paciente masculino, mayor de 60 años, con enfermedad coronaria confirmada"),
    ("Metas de c-LDL", "¿Qué metas de c-LDL se manejan en prevención secundaria?"),
]

st.set_page_config(
    page_title="Agente RAG clínico | CUN",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============ RECURSOS (se cargan una sola vez por sesión) ============

@st.cache_resource(show_spinner="Cargando modelo de embeddings SBERT...")
def modelo():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model_name)


@st.cache_resource(show_spinner=False)
def cliente():
    return get_qdrant_client()


# ============ RENDER ============

def barra_similitud(score: float) -> str:
    pct = max(2.0, min(100.0, score * 100.0))
    clase = "alto" if score >= 0.65 else ""
    return f'<div class="barra"><i class="{clase}" style="width:{pct:.1f}%"></i></div>'


def tarjeta(rank: int, hit: dict) -> str:
    """Fragmento recuperado como hoja del expediente."""
    meta = hit["metadata"]
    fuente = meta.get("title") or f"Historia de Paciente #{meta.get('patient_id', '—')}"
    clinico = ""
    if meta.get("patient_id"):
        clinico = (
            f"<span>edad <b>{meta.get('age', '—')}</b> años</span>"
            f"<span>sexo <b>{meta.get('sex', '—')}</b></span>"
            f"<span>dolor <b>{meta.get('chest_pain_label', '—')}</b></span>"
            f'<span class="r-riesgo">riesgo <b>{meta.get("risk_level", "—")}</b></span>'
        )
    html = "".join([
        '<div class="resultado">',
        f'<div class="r-cab"><span class="r-puesto">FUENTE {rank} · {meta.get("doc_type", "—")}</span>'
        f'<span class="r-fuente">{fuente}</span></div>',
        f'<div class="r-meta"><span>similitud coseno <b>{hit["score"]:.4f}</b></span>{clinico}</div>',
        barra_similitud(hit["score"]),
        f'<div class="r-texto">{_MD.render(hit["text"][:2000])}</div>',
        "</div>",
    ])
    # shortcut: un <pre> con líneas en blanco dentro del chunk se aplana;
    # subir a un render fragmento a fragmento si el corpus usa <pre>.
    # Una línea en blanco dentro del HTML corta el bloque de markdown y el
    # resto se imprime como texto literal.
    return "\n".join(linea for linea in html.splitlines() if linea.strip())


def responder(pregunta: str) -> dict:
    """Consulta el clúster y devuelve el mensaje del agente ya renderizable."""
    # Contexto conversacional: turnos previos (sin el prompt actual) para
    # seguir conversando. La búsqueda semántica se ayuda de la pregunta
    # anterior cuando el seguimiento es corto o anafórico.
    previos = st.session_state.mensajes[:-1]
    historial = []
    for m in previos:
        if m["role"] == "user":
            historial.append({"role": "user", "content": m["content"]})
        elif m.get("texto"):
            historial.append({"role": "assistant", "content": m["texto"]})
    pregunta_anterior = next(
        (m["content"] for m in reversed(previos) if m["role"] == "user"), ""
    )
    consulta_busqueda = (
        f"{pregunta_anterior} {pregunta}".strip() if pregunta_anterior else pregunta
    )

    parseado = parse_query(pregunta)
    hits = semantic_search(
        cliente(),
        modelo(),
        consulta_busqueda,
        collection_name=settings.collection_name,
        limit=TOP_K,
        query_filter=parseado.to_filter(),
    )

    if not hits:
        return {
            "role": "assistant",
            "html": '<div class="lead">No encontré información para esa consulta con las '
                    "restricciones detectadas. Intenta reformularla o quitar el filtro "
                    "implícito (edad, sexo, diagnóstico).</div>",
        }

    # PASO generativo: el modelo remoto redacta UNA sola respuesta natural y
    # estructurada a partir de los fragmentos recuperados. Sin fallback
    # extractivo: si la API falla se muestra un error claro y listo.
    try:
        respuesta = generar_respuesta(pregunta, hits, historial=historial)
    except Exception as exc:
        return {
            "role": "assistant",
            "html": (
                "<div class='lead'>El modelo generativo no pudo redactar la "
                f"respuesta ahora mismo ({exc}). Reintenta la consulta.</div>"
            ),
        }

    cuerpo = f'<div class="r-texto r-respuesta">{_MD.render(respuesta)}</div>'
    return {
        "role": "assistant",
        "html": cuerpo,
        "texto": respuesta,
        "fuentes": [tarjeta(i, hit) for i, hit in enumerate(hits, 1)],
    }


def bienvenida():
    with st.chat_message("assistant"):
        st.markdown(
            '<div class="lead">Hola. Soy el agente RAG del proyecto: busco por significado en '
            "<b>303 historias clínicas</b> y <b>3 guías normativas</b> indexadas en "
            "<b>Qdrant Cloud</b> con embeddings SBERT de 384 dimensiones. "
            "Pregúntame por criterios diagnósticos, metas de c-LDL, riesgo cardiovascular "
            "o características de un paciente.</div>",
            unsafe_allow_html=True,
        )
        st.caption("Si tu pregunta menciona edad, sexo, diagnóstico o riesgo, el sistema "
                   "lo aplica como filtro de metadatos. Prueba una consulta:")
        for col, (etiqueta, texto) in zip(st.columns(len(EJEMPLOS)), EJEMPLOS):
            if col.button(etiqueta, width="stretch"):
                st.session_state.pendiente = texto
                st.rerun()


def barra_lateral():
    with st.sidebar:
        st.markdown(
            f"""
            <div class="marca-app">
                {icono("corazon", "ico ico-lg")}
                <div>
                    <div class="t1">Agente RAG clínico</div>
                    <div class="t2">Demo ACA N° 1 · CUN</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Nueva conversación",
            width="stretch",
            on_click=lambda: st.session_state.update(mensajes=[]),
        )
        st.markdown(
            """
            <div class="pie-indice">
                <b>Corpus</b> · 303 historias UCI Heart Disease + 3 guías clínicas<br>
                <b>Índice</b> · Qdrant Cloud (AWS) · SBERT 384-d<br>
                <b>Agente</b> · recupera fragmentos reales y los sintetiza con
                {settings.generative_model_name} (API remota) en una única
                respuesta natural y estructurada
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("mensajes", [])
    barra_lateral()

    prompt = st.chat_input("Pregunta sobre las guías clínicas o las historias de pacientes...")
    prompt = prompt or st.session_state.pop("pendiente", None)

    if prompt:
        st.session_state.mensajes.append({"role": "user", "content": prompt})
        with st.spinner("Buscando en Qdrant Cloud y redactando la respuesta con el modelo generativo..."):
            try:
                st.session_state.mensajes.append(responder(prompt))
            except Exception as exc:
                st.session_state.mensajes.append({
                    "role": "assistant",
                    "error": f"No pude consultar el clúster. Verifica las variables "
                             f"`QDRANT_URL` / `QDRANT_API_KEY` del `.env`. Detalle: {exc}",
                })

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            if "error" in mensaje:
                st.error(mensaje["error"])
            elif mensaje["role"] == "user":
                st.markdown(mensaje["content"])
            else:
                st.markdown(mensaje["html"], unsafe_allow_html=True)
                if mensaje.get("fuentes"):
                    with st.expander(f"Ver las {len(mensaje['fuentes'])} fuentes"):
                        st.markdown("".join(mensaje["fuentes"]), unsafe_allow_html=True)

    if not st.session_state.mensajes:
        bienvenida()


if __name__ == "__main__":
    main()
