"""
Tests estrictos del parser de consultas (PASO 5, refinamiento).

Se cubren los escenarios que importan: detección correcta, **no** detección
cuando la restricción no está escrita, casos trampa (negativos que contienen
positivos), ambigüedades y construcción del filtro de Qdrant.
"""

from __future__ import annotations

import pytest

from rag.query_parser import (
    ParsedQuery,
    describe_filter_conditions,
    merge_conditions,
    parse_query,
)

# --------------------------------------------------------------------------- #
# Escenario 1 — la consulta de referencia de la sustentación
# --------------------------------------------------------------------------- #

CONSULTA_EDITO = "paciente masculino, mayor de 60 años, con enfermedad coronaria confirmada"


def test_consulta_de_referencia_detecta_todas_las_restricciones():
    p = parse_query(CONSULTA_EDITO)
    assert p.doc_type == "patient_record"
    assert p.sex == "Masculino"
    assert p.min_age == 60
    assert p.max_age is None
    assert p.disease is True
    assert p.risk is None
    assert p.has_filters
    assert p.conditions == [
        "tipo=patient_record",
        "sexo=Masculino",
        "edad>=60",
        "enfermedad=confirmada",
    ]


def test_la_consulta_original_no_se_modifica():
    p = parse_query(CONSULTA_EDITO)
    assert p.text == CONSULTA_EDITO


# --------------------------------------------------------------------------- #
# Escenario 2 — detección de edad (todas las variantes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("paciente mayor de 60 años", 60),
        ("paciente mas de 60 años", 60),
        ("paciente de más de 65 años", 65),
        ("pacientes mayores de 70 años", 70),
        ("paciente de 60 años o más", 60),
        ("paciente de 55 años o mas", 55),
        ("paciente >60 años", 60),
        ("paciente de 80 años en adelante", 80),
        ("a partir de 50 años", 50),
        ("de 68 años en adelante", 68),
    ],
)
def test_detecta_edad_minima_en_todas_sus_variantes(texto, esperado):
    assert parse_query(texto).min_age == esperado


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("paciente menor de 50 años", 50),
        ("paciente menos de 45 años", 45),
        ("paciente <50 años", 50),
        ("paciente de 50 años o menos", 50),
        ("paciente hasta 40 años", 40),
    ],
)
def test_detecta_edad_maxima_en_todas_sus_variantes(texto, esperado):
    assert parse_query(texto).max_age == esperado


def test_no_detecta_edad_si_no_se_escribe():
    p = parse_query("paciente masculino con angina típica")
    assert p.min_age is None
    assert p.max_age is None


def test_rango_contradictorio_se_descarta():
    p = parse_query("paciente mayor de 70 años y menor de 40 años")
    assert p.min_age is None
    assert p.max_age is None
    assert any("contradictorio" in nota for nota in p.notes)


def test_toma_el_maximo_de_las_edades_minimas_detectadas():
    p = parse_query("paciente mayor de 50 años, de 60 años o más")
    assert p.min_age == 60


# --------------------------------------------------------------------------- #
# Escenario 3 — detección de sexo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("paciente masculino", "Masculino"),
        ("pacientes masculinos", "Masculino"),
        ("hombre de 60 años", "Masculino"),
        ("hombres con angina", "Masculino"),
        ("varón con cardiopatía", "Masculino"),
        ("paciente femenina", "Femenino"),
        ("mujer de 70 años", "Femenino"),
        ("mujeres con angina", "Femenino"),
    ],
)
def test_detecta_sexo(texto, esperado):
    assert parse_query(texto).sex == esperado


def test_sexo_ambiguo_no_aplica_filtro():
    p = parse_query("pacientes hombres y mujeres con angina")
    assert p.sex is None
    assert any("ambos sexos" in nota for nota in p.notes)


def test_no_detecta_sexo_si_no_se_escribe():
    assert parse_query("paciente con angina típica").sex is None


# --------------------------------------------------------------------------- #
# Escenario 4 — detección de enfermedad (incluidos los casos trampa)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("paciente con enfermedad coronaria confirmada", True),
        ("paciente con cardiopatía isquémica confirmada", True),
        ("pacientes con angina típica y diagnostico de isquemia", True),
        ("paciente con enfermedad coronaria", True),
        ("paciente con isquemia miocardica confirmada", True),
    ],
)
def test_detecta_enfermedad_confirmada(texto, esperado):
    assert parse_query(texto).disease is True


@pytest.mark.parametrize(
    "texto",
    [
        "paciente sin enfermedad coronaria",
        "paciente sin cardiopatía",
        "paciente sano con angina leve",
        "pacientes no diagnosticados de cardiopatía",
        "paciente con ausencia de enfermedad coronaria",
    ],
)
def test_detecta_enfermedad_ausente(texto):
    assert parse_query(texto).disease is False


def test_caso_trampa_sin_enfermedad_gana_a_confirmada():
    """'sin enfermedad coronaria' contiene 'enfermedad coronaria': debe ser False."""
    assert parse_query("paciente sin enfermedad coronaria").disease is False


def test_no_detecta_enfermedad_si_no_se_escribe():
    assert parse_query("paciente de 60 años con angina típica").disease is None


# --------------------------------------------------------------------------- #
# Escenario 5 — tipo de documento
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("paciente con angina", "patient_record"),
        ("historia clínica del paciente", "patient_record"),
        ("criterios de angina típica", "clinical_guideline"),
        ("dolor torácico en la consulta", "clinical_guideline"),
        ("estratificación del riesgo coronario", "clinical_guideline"),
        ("ergometría y prueba de esfuerzo", "diagnostic_protocol"),
        ("depresión del segmento ST en el esfuerzo", "diagnostic_protocol"),
        ("angiografía coronaria", "diagnostic_protocol"),
        ("gammagrafía con spect", "diagnostic_protocol"),
        ("metas de ldl con estatinas", "pharmacology_protocol"),
        ("esquema dapt con ticagrelor", "pharmacology_protocol"),
        ("prevención secundaria", "pharmacology_protocol"),
    ],
)
def test_detecta_tipo_de_documento(texto, esperado):
    assert parse_query(texto).doc_type == esperado


def test_paciente_tiene_prioridad_sobre_angina():
    """'paciente con angina' debe filtrar historias clínicas, no guías."""
    assert parse_query("paciente con angina típica").doc_type == "patient_record"


def test_no_detecta_tipo_si_no_se_escribe():
    assert parse_query("¿cuál es el tratamiento?").doc_type is None


# --------------------------------------------------------------------------- #
# Escenario 6 — nivel de riesgo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("pacientes de riesgo muy alto", "Muy Alto / Crítico"),
        ("pacientes con riesgo alto", "Alto"),
        ("paciente de alto riesgo cardiovascular", "Alto"),
        ("riesgo moderado en la consulta", "Moderado"),
        ("pacientes de riesgo bajo", "Bajo"),
    ],
)
def test_detecta_nivel_de_riesgo(texto, esperado):
    assert parse_query(texto).risk == esperado


def test_muy_alto_tiene_prioridad_sobre_alto():
    assert parse_query("riesgo muy alto").risk == "Muy Alto / Crítico"


def test_no_detecta_riesgo_si_no_se_escribe():
    assert parse_query("paciente con angina").risk is None


# --------------------------------------------------------------------------- #
# Escenario 7 — construcción del filtro de Qdrant
# --------------------------------------------------------------------------- #


def test_filtro_constructor_de_la_consulta_de_referencia():
    filtro = parse_query(CONSULTA_EDITO).to_filter()
    assert filtro is not None
    assert len(filtro.must) == 4
    descripcion = describe_filter_conditions(filtro)
    for esperado in ("doc_type=patient_record", "sex=Masculino", "age>=60", "disease_present=True"):
        assert esperado in descripcion


def test_filtro_none_sin_restricciones():
    assert parse_query("¿qué es un infarto?").to_filter() is None


def test_filtro_no_inventa_un_rango_de_edad_no_escrito():
    """'entre 50 y 70' no es una regla soportada: no debe inventar restricciones de edad."""
    p = parse_query("paciente entre 50 y 70 años")
    assert p.min_age is None
    assert p.max_age is None
    filtro = p.to_filter()
    # Sí detecta el tipo de documento, pero el filtro NO debe acotar la edad.
    descripcion = describe_filter_conditions(filtro)
    assert "age" not in descripcion
    assert "doc_type=patient_record" in descripcion


def test_los_filtros_manuales_tienen_prioridad_sobre_los_detectados():
    p = parse_query(CONSULTA_EDITO)
    filtro = p.to_filter(sex="Femenino", min_age=30)
    descripcion = describe_filter_conditions(filtro)
    assert "sex=Femenino" in descripcion
    assert "sex=Masculino" not in descripcion
    assert "age>=30" in descripcion
    assert "age>=60" not in descripcion


def test_un_argumento_manual_explicito_none_elimina_la_restriccion():
    p = parse_query(CONSULTA_EDITO)
    filtro = p.to_filter(sex=None)
    descripcion = describe_filter_conditions(filtro)
    assert "sex=" not in descripcion
    assert "doc_type=patient_record" in descripcion


def test_filtro_con_enfermedad_ausente():
    filtro = parse_query("paciente sin enfermedad coronaria").to_filter()
    assert "disease_present=False" in describe_filter_conditions(filtro)


# --------------------------------------------------------------------------- #
# Escenario 8 — combinación con los filtros manuales de la interfaz
# --------------------------------------------------------------------------- #


def test_los_manuales_reemplazan_a_los_detectados_por_clave():
    resultado = merge_conditions(
        detectados=["tipo=patient_record", "sexo=Masculino", "edad>=60"],
        manuales=["sexo=Femenino", "edad>=70"],
    )
    assert "sexo=Femenino" in resultado
    assert "sexo=Masculino" not in resultado
    assert "edad>=70" in resultado
    assert "edad>=60" not in resultado
    assert "tipo=patient_record" in resultado


def test_manuales_vacios_conserva_los_detectados():
    resultado = merge_conditions(detectados=["tipo=patient_record"], manuales=[])
    assert resultado == ["tipo=patient_record"]


# --------------------------------------------------------------------------- #
# Escenario 9 — robustez y casos límite
# --------------------------------------------------------------------------- #


def test_texto_vacio_no_falla():
    p = parse_query("")
    assert not p.has_filters
    assert p.to_filter() is None


def test_texto_con_espacios_excesivos_se_normaliza():
    p = parse_query("   PACIENTE   MASCULINO   MAYOR DE   60 AÑOS   ")
    assert p.doc_type == "patient_record"
    assert p.sex == "Masculino"
    assert p.min_age == 60


def test_mayusculas_y_tildes_se_manejan():
    p = parse_query("PACIENTE MASCULINO MAYOR DE 60 AÑOS CON ENFERMEDAD CORONARIA CONFIRMADA")
    assert p.sex == "Masculino"
    assert p.min_age == 60
    assert p.disease is True


def test_consulta_sin_relacion_con_el_dominio_no_inventa_filtros():
    p = parse_query("¿cuánto cuesta un ithaka en colombia?")
    assert not p.has_filters


def test_explicacion_es_legible():
    p = parse_query(CONSULTA_EDITO)
    texto = p.explanation()
    assert "tipo=patient_record" in texto
    assert "sexo=Masculino" in texto
    assert "edad>=60" in texto


def test_explicacion_sin_filtros():
    assert "No se detectaron" in parse_query("hola").explanation()


def test_parsed_query_es_inmutable():
    p = parse_query(CONSULTA_EDITO)
    with pytest.raises(Exception):
        p.sex = "Femenino"  # type: ignore[misc]


def test_query_parser_no_depende_de_la_red_ni_de_modelos():
    """El parser debe ser puramente local: sin red, sin modelos, sin coste."""
    import ast
    import inspect

    import rag.query_parser as modulo

    arbol = ast.parse(inspect.getsource(modulo))
    modulos_importados: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos_importados.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos_importados.add(nodo.module.split(".")[0])

    prohibidos = {"requests", "httpx", "urllib", "urllib3", "aiohttp", "socket"}
    assert not (modulos_importados & prohibidos), f"el parser no debe usar red: {modulos_importados}"
    assert "sentence_transformers" not in modulos_importados
    assert "torch" not in modulos_importados
