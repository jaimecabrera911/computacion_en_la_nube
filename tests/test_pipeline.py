"""
Pruebas del pipeline RAG.

Se ejecutan sin credenciales: cuando no hay `.env`, el pipeline usa el cliente
Qdrant **in-memory**, de modo que las pruebas son rápidas y deterministas.

    uv run pytest
"""

from __future__ import annotations

import pandas as pd

from rag.dataset_loader import compute_clinical_risk, generate_patient_corpus
from rag.evaluation import BENCHMARK_QUERIES, summarize_results
from rag.vector_store import create_or_recreate_collection, index_corpus_in_batches, semantic_search


def _paciente_falso() -> pd.Series:
    return pd.Series(
        {
            "age": 65,
            "sex": 1,
            "cp": 1,
            "trestbps": 140,
            "chol": 250,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 2,
            "ca": 2.0,
            "thal": 7.0,
            "target": 4,
        }
    )


def test_estratificacion_riesgo_escalonada():
    """El nivel de riesgo debe escalar con la severidad de los hallazgos."""
    suave = _paciente_falso().copy()
    suave.update({"target": 0, "ca": 0.0, "oldpeak": 0.5, "exang": 0, "thal": 3.0,
                  "trestbps": 120, "chol": 190, "fbs": 0})
    assert compute_clinical_risk(suave) == "Bajo"

    moderado = suave.copy()
    moderado["trestbps"] = 150
    assert compute_clinical_risk(moderado) == "Moderado"

    alto = suave.copy()
    alto["thal"] = 7.0
    assert compute_clinical_risk(alto) == "Alto"

    critico = suave.copy()
    critico.update({"target": 3, "ca": 2.0})
    assert compute_clinical_risk(critico) == "Muy Alto / Crítico"


def test_narrativa_clinica_incluye_metadatos_clave():
    """La viñeta clínica debe contener lenguaje natural y metadatos consultables."""
    corpus = generate_patient_corpus(pd.DataFrame([_paciente_falso()]))
    documento = corpus[0]

    assert "HISTORIA CLÍNICA CARDIOLÓGICA" in documento["text"]
    assert "Angina típica" in documento["text"]
    assert "65 años" in documento["text"]

    meta = documento["metadata"]
    assert meta["doc_type"] == "patient_record"
    assert meta["patient_id"] == 1
    assert meta["sex"] == "Masculino"
    assert meta["disease_present"] is True
    assert meta["risk_level"] == "Muy Alto / Crítico"
    assert meta["age_group"] == ">=65"


def test_chunking_respeta_parametros():
    """Los fragmentos no deben exceder el tamaño configurado y deben traer metadatos."""
    from rag.config import settings
    from rag.document_processor import chunk_unstructured_documents

    documentos = [
        {
            "filename": "DOC_TEST.md",
            "format": "markdown",
            "content": "\n\n".join(["Párrafo clínico de prueba. " * 30] * 6),
            "metadata": {"title": "Test", "doc_type": "clinical_guideline"},
        }
    ]
    fragmentos = chunk_unstructured_documents(documentos)

    assert len(fragmentos) > 1
    assert all(f["metadata"]["chunk_char_length"] <= settings.chunk_size for f in fragmentos)
    assert [f["metadata"]["chunk_id"] for f in fragmentos] == list(range(1, len(fragmentos) + 1))


def test_busqueda_semantica_recupera_el_documento_correcto():
    """La búsqueda debe recuperar el fragmento del documento semánticamente más cercano."""
    from sentence_transformers import SentenceTransformer

    from rag.config import settings
    from rag.vector_store import get_qdrant_client

    client = get_qdrant_client()
    create_or_recreate_collection(client, collection_name="test_collection", vector_size=384)

    modelo = SentenceTransformer(settings.embedding_model_name)
    corpus = [
        {
            "text": "La angina típica se caracteriza por dolor opresivo retroesternal inducido por el esfuerzo.",
            "metadata": {"doc_type": "clinical_guideline", "title": "Guía de angina"},
        },
        {
            "text": "La diabetes mellitus tipo 2 se controla con metformina y dieta hipocalórica.",
            "metadata": {"doc_type": "clinical_guideline", "title": "Guía de diabetes"},
        },
    ]
    index_corpus_in_batches(client, modelo, corpus, collection_name="test_collection", batch_size=8)

    resultados = semantic_search(
        client=client,
        model=modelo,
        query="dolor en el pecho opresivo al hacer ejercicio",
        collection_name="test_collection",
        limit=1,
    )
    assert resultados[0]["metadata"]["title"] == "Guía de angina"


def test_bateria_de_consultas_es_consistente():
    """La batería oficial debe estar bien formada y ser serializable."""
    assert len(BENCHMARK_QUERIES) == 5
    ids = [q["id"] for q in BENCHMARK_QUERIES]
    assert ids == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for consulta in BENCHMARK_QUERIES:
        # Las consultas pueden ser interrogativas o escenarios clínicos declarativos.
        assert len(consulta["query"].strip()) > 30
        assert "filter_label" in consulta
        assert "expected" in consulta


def test_resumen_de_metricas_calcula_promedios():
    """El resumen agregado debe derivar sus valores del detalle Top-k."""
    detalle = pd.DataFrame(
        [
            {"query_id": "Q1", "rank": 1, "score": 0.6, "doc_type": "clinical_guideline",
             "source_title": "A", "snippet": "x"},
            {"query_id": "Q1", "rank": 2, "score": 0.5, "doc_type": "clinical_guideline",
             "source_title": "B", "snippet": "y"},
        ]
    )
    resumen = summarize_results(detalle)
    assert resumen["top1_mean"] == 0.6
    assert resumen["top1_max"] == 0.6
    assert resumen["by_rank"]["1"] == 0.6
    assert resumen["by_rank"]["2"] == 0.5
