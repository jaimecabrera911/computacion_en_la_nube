"""
PASO 5 (refinamiento) — Extracción automática de filtros de metadatos desde la consulta.

Por qué existe este módulo
--------------------------
Un modelo de embeddings **no es un modelo de lenguaje que razona**: es una función
``texto -> vector``. No puede comparar números ni seguir instrucciones. Por eso pedir
*"mayor de 60 años"* dentro del texto de una consulta **no** restringe la edad de los
resultados: esa frase solo aporta una dirección semántica, no una condición.

La solución que aplican los sistemas RAG reales es separar dos responsabilidades:

    pregunta en lenguaje natural
            ↓
    [extractor de filtros]  ← este módulo (determinista, sin costo, sin red)
            ↓  {doc_type, sexo, edad, enfermedad, riesgo}
    base vectorial: búsqueda por similitud CON el filtro aplicado
            ↓
    Top-k resultados

Este parser es **determinista** (listas de palabras clave y expresiones regulares),
**explicable** (devuelve qué detectó y por qué) y **gratuito** (no consume tokens
ni cuota de API). La alternativa con un LLM se reserva para el paso siguiente del
proyecto: redactar la respuesta a partir de los fragmentos recuperados.

Regla de oro: **solo se aplica lo que la consulta menciona explícitamente**.
Nunca se infiere una restricción que el usuario no escribió.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from qdrant_client.http import models

#: Valor centinela para distinguir "no especificado" de "especificado como None".
UNSET: Any = object()

# --------------------------------------------------------------------------- #
# Listas de palabras clave
# --------------------------------------------------------------------------- #

#: Orden = prioridad. `patient_record` va primero porque es el filtro más restrictivo:
#: si la consulta habla de "paciente", el resultado buscado es una historia clínica,
#: aunque también mencione "angina" (que pertenece a la guía clínica).
DOC_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "patient_record",
        [
            "paciente",
            "pacientes",
            "historia clinica",
            "historias clinicas",
            "expediente clinico",
            "expedientes clinicos",
        ],
    ),
    (
        "diagnostic_protocol",
        [
            "ergometria",
            "prueba de esfuerzo",
            "pruebas de esfuerzo",
            "prueba de esfuerzo fisica",
            "segmento st",
            "electrocardiograma de esfuerzo",
            "angiografia",
            "gammagrafia",
            "spect",
            "eco de esfuerzo",
        ],
    ),
    (
        "pharmacology_protocol",
        [
            "estatina",
            "estatinas",
            "farmacologico",
            "antiagregacion",
            "dapt",
            "ldl",
            "c-ldl",
            "ticagrelor",
            "prevencion secundaria",
            "medicamentacion",
        ],
    ),
    (
        "clinical_guideline",
        [
            "angina",
            "dolor toracico",
            "guia clinica",
            "estratificacion",
            "dolor precordial",
        ],
    ),
]

MALE_TERMS = ["masculino", "masculinos", "hombre", "hombres", "varon", "varones"]
FEMALE_TERMS = [
    "femenino",
    "femeninos",
    "femenina",
    "femeninas",
    "mujer",
    "mujeres",
    "senora",
    "senoras",
]

RISK_PATTERNS: list[tuple[str, str]] = [
    ("Muy Alto / Crítico", r"(?:riesgo\s+muy\s+alto|muy\s+alto\s+riesgo)"),
    ("Alto", r"(?:riesgo\s+alto|alto\s+riesgo)"),
    ("Moderado", r"(?:riesgo\s+moderado|moderado\s+riesgo)"),
    ("Bajo", r"(?:riesgo\s+bajo|bajo\s+riesgo)"),
]

# Edades mínimas: "mayor de 60", "más de 60", ">60", "60 años o más", "a partir de 60".
MIN_AGE_PATTERNS = [
    r"(?:mayor(?:es)?|mas)\s+de\s+(\d{1,3})",
    r">\s*(\d{1,3})\b",
    r"(\d{1,3})\s*(?:anos\s*)?o\s+mas",
    r"a\s+partir\s+de\s+(\d{1,3})",
    r"(\d{1,3})\s*anos\s+en\s+adelante",
    r"de\s+(\d{1,3})\s*anos\s+(?:en\s+)?adelante",
]

# Edades máximas: "menor de 50", "menos de 50", "<50", "50 años o menos".
MAX_AGE_PATTERNS = [
    r"(?:menor(?:es)?|menos)\s+de\s+(\d{1,3})",
    r"<\s*(\d{1,3})\b",
    r"(\d{1,3})\s*(?:anos\s*)?o\s+menos",
    r"hasta\s+(\d{1,3})\s*anos",
]

# Negativos primero: "sin enfermedad" contiene la palabra "enfermedad".
DISEASE_FALSE_PATTERNS = [
    r"sin\s+enfermedad",
    r"sin\s+cardiopatia",
    r"ausencia\s+de\s+enfermedad",
    r"no\s+diagnosticad[oa]s?",
    r"\bsanos?\b",
    r"sin\s+isquemia",
]

DISEASE_TRUE_PATTERNS = [
    r"enfermedad\s+coronaria\s+confirmada",
    r"enfermedad\s+confirmada",
    r"cardiopatia\s+(?:isquemica\s+)?confirmada",
    r"con\s+enfermedad\s+coronaria",
    r"diagnosticad[oa]s?",
    r"diagnostico\s+de",
    r"enfermedad\s+coronaria",
    r"isquemia\s+miocardica\s+confirmada",
]


def _normalize(text: str) -> str:
    """
    Minúsculas, espacios colapsados y **tildes eliminadas**.

    Quitar las tildes hace que el parser sea indiferente a la ortografía:
    *'varón'*, *'varon'* y *'VARON'* se reconocen igual, y las expresiones
    regulares se escriben una sola vez.
    """
    sin_tildes = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", text)
        if unicodedata.category(caracter) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_tildes.strip().lower())


@dataclass(frozen=True)
class ParsedQuery:
    """Resultado del análisis de una consulta en lenguaje natural."""

    text: str
    doc_type: str | None = None
    sex: str | None = None
    min_age: int | None = None
    max_age: int | None = None
    disease: bool | None = None
    risk: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- Estado ---------------------------------------------------------------

    @property
    def has_filters(self) -> bool:
        """True si se detectó al menos una restricción aplicable."""
        return any(
            valor is not None
            for valor in (self.doc_type, self.sex, self.min_age, self.max_age, self.disease, self.risk)
        )

    @property
    def conditions(self) -> list[str]:
        """Restricciones detectadas, en texto legible."""
        items: list[str] = []
        if self.doc_type:
            items.append(f"tipo={self.doc_type}")
        if self.sex:
            items.append(f"sexo={self.sex}")
        if self.min_age is not None:
            items.append(f"edad>={self.min_age}")
        if self.max_age is not None:
            items.append(f"edad<={self.max_age}")
        if self.disease is not None:
            items.append("enfermedad=confirmada" if self.disease else "enfermedad=ausente")
        if self.risk:
            items.append(f"riesgo={self.risk}")
        return items

    # -- Traducción a filtro de Qdrant ----------------------------------------

    def to_filter(
        self,
        *,
        doc_type: Any = UNSET,
        sex: Any = UNSET,
        min_age: Any = UNSET,
        max_age: Any = UNSET,
        disease: Any = UNSET,
        risk: Any = UNSET,
    ) -> models.Filter | None:
        """
        Convierte el análisis en un filtro de payload de Qdrant.

        Los argumentos explícitos **tienen prioridad** sobre lo detectado: así, si
        el usuario fija un filtro manual en la interfaz, ese manda.
        """
        resolved = {
            "doc_type": self.doc_type if doc_type is UNSET else doc_type,
            "sex": self.sex if sex is UNSET else sex,
            "min_age": self.min_age if min_age is UNSET else min_age,
            "max_age": self.max_age if max_age is UNSET else max_age,
            "disease": self.disease if disease is UNSET else disease,
            "risk": self.risk if risk is UNSET else risk,
        }

        condiciones: list[models.Condition] = []
        if resolved["doc_type"]:
            condiciones.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=resolved["doc_type"]))
            )
        if resolved["sex"]:
            condiciones.append(
                models.FieldCondition(key="sex", match=models.MatchValue(value=resolved["sex"]))
            )
        if resolved["disease"] is not None:
            condiciones.append(
                models.FieldCondition(
                    key="disease_present", match=models.MatchValue(value=bool(resolved["disease"]))
                )
            )
        if resolved["min_age"] is not None or resolved["max_age"] is not None:
            condiciones.append(
                models.FieldCondition(
                    key="age",
                    range=models.Range(
                        gte=resolved["min_age"],
                        lte=resolved["max_age"],
                    ),
                )
            )
        if resolved["risk"]:
            condiciones.append(
                models.FieldCondition(key="risk_level", match=models.MatchValue(value=resolved["risk"]))
            )

        if not condiciones:
            return None
        return models.Filter(must=condiciones)

    # -- Presentación ----------------------------------------------------------

    def explanation(self) -> str:
        """Resumen legible de lo detectado, para mostrar en la interfaz."""
        if not self.has_filters:
            return "No se detectaron restricciones en la consulta (búsqueda sin filtro)."
        partes = " · ".join(self.conditions)
        extra = ""
        if self.notes:
            extra = f"  (notas: {'; '.join(self.notes)})"
        return f"{partes}{extra}"


# --------------------------------------------------------------------------- #
# Detección
# --------------------------------------------------------------------------- #


def _detect_doc_type(text: str) -> str | None:
    """Primer tipo de documento que coincida, respetando el orden de prioridad."""
    for doc_type, keywords in DOC_TYPE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return doc_type
    return None


def _detect_sex(text: str) -> tuple[str | None, list[str]]:
    """Detecta el sexo. Si se mencionan ambos, no se aplica filtro (es ambiguo)."""
    notas: list[str] = []
    hay_macho = any(term in text for term in MALE_TERMS)
    hay_hembra = any(term in text for term in FEMALE_TERMS)
    if hay_macho and hay_hembra:
        notas.append("se mencionan ambos sexos: no se aplica filtro de sexo")
        return None, notas
    if hay_macho:
        return "Masculino", notas
    if hay_hembra:
        return "Femenino", notas
    return None, notas


def _detect_ages(text: str) -> tuple[int | None, int | None, list[str]]:
    """Detecta la edad mínima y la máxima. Un rango contradictorio se descarta."""
    notas: list[str] = []

    min_candidates = [
        int(m.group(1))
        for patron in MIN_AGE_PATTERNS
        for m in re.finditer(patron, text)
    ]
    max_candidates = [
        int(m.group(1))
        for patron in MAX_AGE_PATTERNS
        for m in re.finditer(patron, text)
    ]

    edad_min = max(min_candidates) if min_candidates else None
    edad_max = min(max_candidates) if max_candidates else None

    if edad_min is not None and edad_max is not None and edad_min > edad_max:
        notas.append(
            f"rango de edad contradictorio (>{edad_min} y <{edad_max}): se descarta el rango"
        )
        edad_min = edad_max = None

    return edad_min, edad_max, notas


def _detect_disease(text: str) -> bool | None:
    """Detecta si se pide enfermedad confirmada o ausente (los negativos van primero)."""
    for patron in DISEASE_FALSE_PATTERNS:
        if re.search(patron, text):
            return False
    for patron in DISEASE_TRUE_PATTERNS:
        if re.search(patron, text):
            return True
    return None


def _detect_risk(text: str) -> str | None:
    """Detecta el nivel de riesgo solicitado (se evalúa 'muy alto' antes que 'alto')."""
    for nivel, patron in RISK_PATTERNS:
        if re.search(patron, text):
            return nivel
    return None


def parse_query(text: str) -> ParsedQuery:
    """
    Analiza una consulta en lenguaje natural y extrae las restricciones estructuradas.

    La consulta original **no se modifica**: el texto completo se sigue vectorizando
    (el modelo se beneficia del contexto completo), y las restricciones se aplican
    además como filtro de payload.

    Args:
        text: pregunta del usuario en lenguaje natural.

    Returns:
        ``ParsedQuery`` con las restricciones detectadas y su explicación.
    """
    normalizado = _normalize(text)

    doc_type = _detect_doc_type(normalizado)
    sex, notas = _detect_sex(normalizado)
    min_age, max_age, notas_edad = _detect_ages(normalizado)
    disease = _detect_disease(normalizado)
    risk = _detect_risk(normalizado)

    return ParsedQuery(
        text=text,
        doc_type=doc_type,
        sex=sex,
        min_age=min_age,
        max_age=max_age,
        disease=disease,
        risk=risk,
        notes=tuple(notas + notas_edad),
    )


def describe_filter_conditions(filtro: models.Filter | None) -> str:
    """Describe un filtro de Qdrant en texto legible (para la interfaz)."""
    if filtro is None:
        return "sin filtro"
    partes: list[str] = []
    for condicion in filtro.must:
        if isinstance(condicion, models.FieldCondition):
            clave = condicion.key
            if condicion.match is not None:
                partes.append(f"{clave}={condicion.match.value}")
            elif condicion.range is not None:
                if condicion.range.gte is not None:
                    partes.append(f"{clave}>={int(condicion.range.gte)}")
                if condicion.range.lte is not None:
                    partes.append(f"{clave}<={int(condicion.range.lte)}")
    return " · ".join(partes) if partes else "sin filtro"


def merge_conditions(
    detectados: Iterable[str],
    manuales: Iterable[str],
) -> list[str]:
    """
    Combina restricciones detectadas y manuales: **las manuales tienen prioridad**.

    Args:
        detectados: restricciones extraídas del texto de la consulta.
        manuales: restricciones fijadas explícitamente por el usuario en la interfaz.

    Returns:
        Lista final de restricciones, sin duplicados y en orden estable.
    """
    resultado = list(detectados)
    for manual in manuales:
        clave = manual.split("=", 1)[0]
        resultado = [c for c in resultado if not c.startswith(f"{clave}=")]
        resultado.append(manual)
    return resultado
