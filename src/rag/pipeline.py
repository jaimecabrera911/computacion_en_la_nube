"""
Orquestador del pipeline RAG (PASO 1 → PASO 5 de la ruta de aprendizaje del ACA 1).

Este módulo concentra **toda** la lógica del pipeline en un único lugar para que el
notebook, la CLI y la app de sustentación compartan exactamente el mismo
comportamiento (una sola fuente de verdad, sin código duplicado).

Flujo completo:

    PASO 1  Aprovisionamiento   → colección vectorial + índices de payload
    PASO 2  Ingesta y chunking  → dataset tabular narrativizado + corpus documental
    PASO 3  Vectorización       → embeddings densos (SBERT, 384 dims, normalizados)
    PASO 4  Indexación batch    → batch upsert al clúster Qdrant Cloud
    PASO 5  Búsqueda semántica  → benchmark Top-k + métricas de similitud coseno
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.dataset_loader import generate_patient_corpus, load_heart_disease_dataset
from rag.document_processor import chunk_unstructured_documents, load_unstructured_documents
from rag.evaluation import (
    BENCHMARK_QUERIES,
    plot_similarity_metrics,
    run_retrieval_benchmark,
    summarize_results,
)
from rag.vector_store import (
    create_or_recreate_collection,
    get_qdrant_client,
    index_corpus_in_batches,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Resumen verificable de una ejecución completa del pipeline."""

    collection_name: str
    mode: str
    patients: int = 0
    document_chunks: int = 0
    indexed_vectors: int = 0
    embedding_seconds: float = 0.0
    search_seconds: float = 0.0
    top1_mean: float = 0.0
    top1_max: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)
    queries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_corpus(self) -> int:
        return self.patients + self.document_chunks

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "mode": self.mode,
            "patients": self.patients,
            "document_chunks": self.document_chunks,
            "total_corpus": self.total_corpus,
            "indexed_vectors": self.indexed_vectors,
            "embedding_seconds": round(self.embedding_seconds, 2),
            "search_seconds": round(self.search_seconds, 2),
            "top1_mean": round(self.top1_mean, 4),
            "top1_max": round(self.top1_max, 4),
            "latency": self.latency,
            "queries": self.queries,
        }


def build_corpus(documents_dir: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """
    PASO 2 — Carga y estructuración de las dos fuentes de datos.

    Returns:
        (viñetas de paciente, fragmentos documentales, dataframe original)
    """
    df_raw, _ = load_heart_disease_dataset()
    logger.info("Dataset UCI Heart Disease cargado: %s pacientes.", len(df_raw))
    patient_corpus = generate_patient_corpus(df_raw)

    docs_dir = documents_dir or str(settings.documents_dir)
    raw_docs = load_unstructured_documents(docs_dir)
    chunked_docs = chunk_unstructured_documents(raw_docs)
    logger.info(
        "Corpus documental cargado: %s documentos -> %s fragmentos.",
        len(raw_docs),
        len(chunked_docs),
    )
    return patient_corpus, chunked_docs, df_raw


def load_embedding_model() -> SentenceTransformer:
    """Carga el modelo de embeddings configurado en `.env`."""
    logger.info("Cargando modelo de embeddings: %s", settings.embedding_model_name)
    return SentenceTransformer(settings.embedding_model_name)


def measure_latency(
    client,
    model: SentenceTransformer,
    top_k: int = 5,
    repetitions: int = 5,
) -> dict[str, float]:
    """
    Mide la latencia real de la búsqueda semántica contra el clúster remoto.

    Se hace una consulta de calentamiento porque la primera llamada paga el
    establecimiento de la conexión TLS y el handshake HTTP; a partir de la
    segunda, la latencia refleja la operación real de búsqueda.
    """
    queries = BENCHMARK_QUERIES
    vectors = [model.encode(bq["query"], normalize_embeddings=True).tolist() for bq in queries]

    # Calentamiento de la conexión.
    client.query_points(collection_name=settings.collection_name, query=vectors[0], limit=1)

    samples: list[float] = []
    for bq, vector in zip(queries, vectors):
        for _ in range(repetitions):
            t0 = time.perf_counter()
            client.query_points(
                collection_name=settings.collection_name,
                query=vector,
                limit=top_k,
                query_filter=bq["filter"],
            )
            samples.append((time.perf_counter() - t0) * 1000)

    # Coste de la vectorización local (SBERT en CPU).
    t0 = time.perf_counter()
    model.encode([bq["query"] for bq in queries], normalize_embeddings=True)
    encode_ms = (time.perf_counter() - t0) * 1000 / len(queries)

    return {
        "search_mean_ms": round(sum(samples) / len(samples), 1),
        "search_median_ms": round(sorted(samples)[len(samples) // 2], 1),
        "search_min_ms": round(min(samples), 1),
        "search_max_ms": round(max(samples), 1),
        "embedding_ms": round(encode_ms, 1),
    }


def describe_collection(client) -> dict[str, Any]:
    """Describe el contenido real de la colección (totales y reparto por tipo)."""
    from qdrant_client.http import models

    collection = client.get_collection(settings.collection_name)
    total = collection.points_count or 0

    por_tipo: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
            limit=512,
            offset=offset,
            with_payload=["doc_type"],
            with_vectors=False,
        )
        for point in points:
            tipo = (point.payload or {}).get("doc_type", "desconocido")
            por_tipo[tipo] = por_tipo.get(tipo, 0) + 1
        if offset is None:
            break

    return {
        "total_vectors": total,
        "patients": por_tipo.get("patient_record", 0),
        "document_chunks": sum(v for k, v in por_tipo.items() if k != "patient_record"),
        "by_doc_type": por_tipo,
    }


def run_pipeline(
    *,
    reindex: bool = True,
    evaluate: bool = True,
    top_k: int = 5,
    save_results: bool = True,
) -> PipelineResult:
    """
    Ejecuta el pipeline completo de extremo a extremo.

    Args:
        reindex: recrea la colección y reindexa el corpus completo.
        evaluate: ejecuta el benchmark semántico y genera los gráficos.
        top_k: número de fragmentos recuperados por consulta.
        save_results: persiste el resumen en ``output/benchmark_summary.json``.
    """
    started = time.perf_counter()

    # PASO 1 — Aprovisionamiento cloud.
    client = get_qdrant_client()
    mode = "cloud" if settings.is_cloud_mode else "local"
    if reindex:
        create_or_recreate_collection(client)

    # PASO 2 — Ingesta, narrativización y chunking.
    patient_corpus, chunked_docs, _ = build_corpus()
    total_corpus = chunked_docs + patient_corpus

    # PASO 3 y 4 — Vectorización e indexación por lotes.
    model = load_embedding_model()
    embedding_start = time.perf_counter()
    if reindex:
        indexed = index_corpus_in_batches(client, model, total_corpus)
    else:
        indexed = len(total_corpus)
    embedding_seconds = time.perf_counter() - embedding_start

    result = PipelineResult(
        collection_name=settings.collection_name,
        mode=mode,
        patients=len(patient_corpus),
        document_chunks=len(chunked_docs),
        indexed_vectors=indexed,
        embedding_seconds=embedding_seconds,
    )

    # PASO 5 — Búsqueda semántica y evaluación cuantitativa.
    if evaluate:
        search_start = time.perf_counter()
        df_eval = run_retrieval_benchmark(client, model, top_k=top_k)
        result.search_seconds = time.perf_counter() - search_start
        summary = summarize_results(df_eval)
        summary["corpus"] = {
            "patients": len(patient_corpus),
            "document_chunks": len(chunked_docs),
            "total_vectors": len(total_corpus),
            "by_doc_type": describe_collection(client).get("by_doc_type", {}),
        }
        summary["collection_name"] = settings.collection_name
        summary["mode"] = mode
        summary["embedding_seconds"] = round(embedding_seconds, 2)
        summary["latency"] = measure_latency(client, model, top_k=top_k)
        result.latency = summary["latency"]
        result.top1_mean = summary["top1_mean"]
        result.top1_max = summary["top1_max"]
        result.queries = summary["queries"]

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        plot_similarity_metrics(df_eval, str(settings.output_dir / "similarity_distribution.png"))
        if save_results:
            df_eval.to_csv(settings.benchmark_csv_path, index=False)
            with open(settings.benchmark_results_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Pipeline completo en %.1f s", time.perf_counter() - started)
    return result
