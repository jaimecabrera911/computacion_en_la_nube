"""
PASO 6 del pipeline RAG — Generación de respuesta natural vía API.

Toma la consulta del usuario y los fragmentos recuperados de Qdrant y pide al
modelo generativo remoto (Harogo, API compatible con OpenAI) que redacte
**una única respuesta coherente y conversacional** que sintetice la evidencia
recuperada, en lugar de devolver los fragmentos crudos.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from rag.config import settings

logger = logging.getLogger(__name__)

_INSTRUCCIONES = (
    "Eres un asistente clínico. Tienes información clínica confiable sobre la "
    "que debes basar tu respuesta: úsala como tu conocimiento, pero NUNCA "
    "hables de ella como si fuera material externo.\n"
    "Reglas estrictas:\n"
    "1. NO menciones jamás 'fragmentos', 'documentos', 'contexto', 'fuentes "
    "recuperadas', 'registros', 'información proporcionada' ni nada similar. "
    "No cites con [1], [2] ni referencias numeradas.\n"
    "2. Redacta como ChatGPT: una respuesta directa en español, en lenguaje "
    "natural, clara y entendible, bien estructurada. Usa un párrafo "
    "introductorio y viñetas o subtítulos breves cuando ayuden; omite lo "
    "redundante.\n"
    "3. Habla de los pacientes y de la evidencia en primera persona del "
    "conocimiento: 'Los pacientes mayores de 60 años...', no 'En los casos "
    "recuperados...'.\n"
    "4. Si la información no alcanza para responder, dilo con franqueza y "
    "explica qué se sabe, sin inventar datos clínicos que no tengas.\n"
    "5. Tu respuesta completa debe empezar EXACTAMENTE con el marcador "
    "'RESPUESTA:' seguido del texto final. Todo lo anterior al marcador se "
    "descarta."
)

_PROMPT = (
    "Información clínica disponible:\n{contexto}\n\n"
    "Pregunta del usuario: {pregunta}\n\n"
    "Escribe ahora tu respuesta, comenzando con 'RESPUESTA:'."
)

_MARCADOR = "RESPUESTA:"
#: Tope de tokens generados: el máximo de salida que la API de Harogo acepta
#: para este modelo, para que ningún resumen estructurado quede cortado.
MAX_NEW_TOKENS = 16384


@lru_cache(maxsize=1)
def _cliente():
    """Crea el cliente OpenAI-compatible una sola vez por proceso."""
    from openai import OpenAI

    if not settings.generative_api_key:
        raise RuntimeError(
            "Falta GENERATIVE_API_KEY en el .env para usar el modelo generativo remoto."
        )
    logger.info(
        "Usando el modelo remoto %s en %s...",
        settings.generative_model_name,
        settings.generative_api_base,
    )
    return OpenAI(
        base_url=settings.generative_api_base,
        api_key=settings.generative_api_key,
    )


def _construir_contexto(hits: list[dict[str, Any]], max_fragmentos: int = 8) -> str:
    """Serializa los fragmentos recuperados numerándolos para poder citarlos."""
    partes: list[str] = []
    for i, hit in enumerate(hits[:max_fragmentos], start=1):
        meta = hit.get("metadata", {})
        fuente = meta.get("title") or (
            f"Historia de paciente #{meta.get('patient_id', '?')}"
            if meta.get("patient_id")
            else "fragmento sin título"
        )
        partes.append(f"[{i}] Fuente: {fuente}\n{hit.get('text', '')[:1500]}")
    return "\n\n".join(partes)


def generar_respuesta(
    pregunta: str,
    hits: list[dict[str, Any]],
    historial: list[dict[str, str]] | None = None,
) -> str:
    """
    Redacta una respuesta natural con el modelo remoto a partir de los fragmentos.

    Args:
        pregunta: consulta original del usuario.
        hits: fragmentos devueltos por ``semantic_search`` (score/text/metadata).
        historial: turnos previos de la conversación como ``{"role", "content"}``
            (solo ``user`` y ``assistant``), para poder hacer preguntas de
            seguimiento con contexto.

    Returns:
        Texto de la respuesta redactada (una sola pieza coherente).

    Raises:
        RuntimeError: si el modelo no devuelve texto para la consulta.
    """
    cliente = _cliente()
    mensajes = [{"role": "system", "content": _INSTRUCCIONES}]
    for turno in (historial or [])[-6:]:
        if turno.get("role") in ("user", "assistant") and turno.get("content"):
            mensajes.append({"role": turno["role"], "content": turno["content"]})
    mensajes.append(
        {
            "role": "user",
            "content": _PROMPT.format(
                contexto=_construir_contexto(hits),
                pregunta=pregunta,
            ),
        },
    )
    respuesta = cliente.chat.completions.create(
        model=settings.generative_model_name,
        messages=mensajes,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        # El modelo es de razonamiento: sin esto gasta el presupuesto de
        # tokens pensando y `content` llega vacío.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    texto = (respuesta.choices[0].message.content or "").strip()
    if _MARCADOR in texto:
        texto = texto.rsplit(_MARCADOR, 1)[1].strip()
    if not texto:
        raise RuntimeError("El modelo no devolvió texto para esta consulta.")
    return texto
