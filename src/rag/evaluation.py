"""
PASO 5 — Validación de la búsqueda semántica y métricas cuantitativas.

Este módulo contiene:

1. ``BENCHMARK_QUERIES``: la batería oficial de consultas en lenguaje natural.
   Es la **única** definición de las consultas del proyecto; el notebook, la CLI,
   el póster y la app de sustentación la consumen, de modo que los números
   publicados en los entregables siempre corresponden a una ejecución real.
2. ``run_retrieval_benchmark``: ejecuta la batería y devuelve el detalle Top-k.
3. ``summarize_results``: agrega el detalle a las métricas que se publican.
4. ``plot_similarity_metrics``: genera la gráfica de distribución de similitud.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from rag.config import settings
from rag.vector_store import semantic_search

# --------------------------------------------------------------------------- #
# Batería oficial de consultas de validación
# --------------------------------------------------------------------------- #


def _doc_type_filter(doc_type: str) -> models.Filter:
    """Filtro por tipo de documento sobre el índice de payload ``doc_type``."""
    return models.Filter(
        must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
    )


BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "id": "Q1",
        "code": "Q1_CLINICAL_GUIDELINE",
        "description": "Criterios de angina típica y estratificación de riesgo en dolor torácico",
        "query": (
            "¿Cuáles son los criterios clínicos para diagnosticar angina típica "
            "y cuál es el algoritmo de estratificación de riesgo en dolor torácico?"
        ),
        "filter": _doc_type_filter("clinical_guideline"),
        "filter_label": "doc_type: 'clinical_guideline'",
        "expected": "Guía MinSalud: angina tipo 1 e indicación de ergometría.",
    },
    {
        "id": "Q2",
        "code": "Q2_ERGOMETRY_DIAGNOSTIC",
        "description": "Significado clínico de la depresión del ST > 2 mm con pendiente descendente",
        "query": (
            "¿Qué significado clínico tiene una depresión del segmento ST mayor a 2 mm "
            "con pendiente plana o descendente durante la prueba de esfuerzo?"
        ),
        "filter": _doc_type_filter("diagnostic_protocol"),
        "filter_label": "doc_type: 'diagnostic_protocol'",
        "expected": "Protocolo ACC/AHA: isquemia severa multivaso (slope 3).",
    },
    {
        "id": "Q3",
        "code": "Q3_PHARMACOLOGY_SECONDARY",
        "description": "Metas de c-LDL con estatinas y doble antiagregación plaquetaria (DAPT)",
        "query": (
            "¿Cuáles son las metas de c-LDL con estatinas de alta potencia y el esquema "
            "de doble antiagregación plaquetaria con aspirina y ticagrelor?"
        ),
        "filter": _doc_type_filter("pharmacology_protocol"),
        "filter_label": "doc_type: 'pharmacology_protocol'",
        "expected": "Protocolo ESC/OMS: meta c-LDL < 55 mg/dL y DAPT por 12 meses.",
    },
    {
        "id": "Q4",
        "code": "Q4_PATIENT_COHORT_HIGH_RISK",
        "description": "Cohorte de pacientes masculinos mayores de 60 años con cardiopatía confirmada",
        "query": (
            "Paciente masculino mayor de 60 años con angina típica, hipertensión estadio 2, "
            "fluoroscopía con vasos mayores comprometidos y alto riesgo cardiovascular"
        ),
        "filter": models.Filter(
            must=[
                models.FieldCondition(key="doc_type", match=models.MatchValue(value="patient_record")),
                models.FieldCondition(key="sex", match=models.MatchValue(value="Masculino")),
                models.FieldCondition(key="disease_present", match=models.MatchValue(value=True)),
                models.FieldCondition(key="age", range=models.Range(gte=60)),
            ]
        ),
        "filter_label": "doc_type: 'patient_record' + sex: 'Masculino' + disease_present: true + age >= 60",
        "expected": "Historia clínica de paciente de alto riesgo del dataset UCI.",
    },
    {
        "id": "Q5",
        "code": "Q5_HYBRID_CROSS_MODAL",
        "description": "Consulta híbrida abierta: isquemia inducible y decisión terapéutica",
        "query": (
            "Tratamiento de revascularización y pronóstico para defecto de perfusión "
            "reversible en gammagrafía SPECT con angina inducida por ejercicio"
        ),
        "filter": None,
        "filter_label": "Sin filtro (consulta cross-modal)",
        "expected": "Correlación entre hallazgo gammagráfico y cateterismo coronario.",
    },
]


# --------------------------------------------------------------------------- #
# Ejecución y métricas
# --------------------------------------------------------------------------- #


def run_retrieval_benchmark(
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """Ejecuta la batería oficial y devuelve el detalle Top-k de cada consulta."""
    records: list[dict[str, Any]] = []

    for bq in BENCHMARK_QUERIES:
        results = semantic_search(
            client=client,
            model=model,
            query=bq["query"],
            collection_name=settings.collection_name,
            limit=top_k,
            query_filter=bq["filter"],
        )

        for rank, res in enumerate(results, start=1):
            metadata = res["metadata"]
            records.append(
                {
                    "query_id": bq["id"],
                    "query_code": bq["code"],
                    "query": bq["query"],
                    "filter_label": bq["filter_label"],
                    "rank": rank,
                    "score": res["score"],
                    "doc_type": metadata.get("doc_type", "unknown"),
                    "source_title": metadata.get(
                        "title", f"Paciente #{metadata.get('patient_id', 'N/A')}"
                    ),
                    "snippet": res["text"][:160],
                }
            )

    return pd.DataFrame(records)


def summarize_results(df_results: pd.DataFrame) -> dict[str, Any]:
    """
    Agrega el detalle Top-k a las métricas que se publican en los entregables.

    Devuelve un diccionario serializable a JSON, que es el insumo del póster,
    de la guía de resultados, del README y de la app de sustentación.
    """
    if df_results.empty:
        return {"top1_mean": 0.0, "top1_max": 0.0, "queries": [], "by_rank": {}}

    top1 = df_results[df_results["rank"] == 1]
    by_rank = (
        df_results.groupby("rank")["score"].mean().round(4).to_dict()
    )

    queries = []
    per_query_stats = {}
    for bq in BENCHMARK_QUERIES:
        subset = top1[top1["query_id"] == bq["id"]]
        if subset.empty:
            continue
        row = subset.iloc[0]
        all_scores = df_results[df_results["query_id"] == bq["id"]]["score"]
        per_query_stats[bq["id"]] = {
            "mean": round(float(all_scores.mean()), 4),
            "std": round(float(all_scores.std(ddof=0)), 4),
            "min": round(float(all_scores.min()), 4),
            "max": round(float(all_scores.max()), 4),
        }
        queries.append(
            {
                "id": bq["id"],
                "code": bq["code"],
                "description": bq["description"],
                "query": bq["query"],
                "filter_label": bq["filter_label"],
                "top1_score": round(float(row["score"]), 4),
                "doc_type": row["doc_type"],
                "source_title": row["source_title"],
                "snippet": row["snippet"],
                "expected": bq["expected"],
            }
        )

    return {
        "top1_mean": round(float(top1["score"].mean()), 4),
        "top1_max": round(float(top1["score"].max()), 4),
        "top1_min": round(float(top1["score"].min()), 4),
        "all_mean": round(float(df_results["score"].mean()), 4),
        "by_rank": {str(k): float(v) for k, v in by_rank.items()},
        "per_query_stats": per_query_stats,
        "queries": queries,
    }


def save_summary(summary: dict[str, Any], path: str | None = None) -> None:
    """Guarda el resumen del benchmark en JSON (fuente única de los números)."""
    target = path or str(settings.benchmark_results_path)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_summary(path: str | None = None) -> dict[str, Any] | None:
    """Carga el resumen del benchmark si existe; ``None`` en caso contrario."""
    target = path or str(settings.benchmark_results_path)
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def plot_similarity_metrics(
    df_results: pd.DataFrame,
    output_path: str = "output/similarity_distribution.png",
) -> None:
    """Genera la gráfica de distribución de similitud coseno usada en el informe."""
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        data=df_results,
        x="query_id",
        y="score",
        ax=axes[0],
        palette="Blues_d",
        hue="query_id",
        legend=False,
    )
    axes[0].set_title(
        "Distribución de Similitud Coseno por Consulta (Top-5)", fontsize=11, fontweight="bold"
    )
    axes[0].set_xlabel("Identificador de Consulta", fontsize=10)
    axes[0].set_ylabel("Puntuación de Similitud Coseno", fontsize=10)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(0.0, 1.0)

    rank_means = df_results.groupby("rank")["score"].mean().reset_index()
    sns.barplot(
        data=rank_means,
        x="rank",
        y="score",
        ax=axes[1],
        palette="crest",
        hue="rank",
        legend=False,
    )
    axes[1].set_title(
        "Similitud Coseno Media según Posición de Recuperación (Rank 1-5)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel("Rango de Recuperación (Top-k)", fontsize=10)
    axes[1].set_ylabel("Similitud Coseno Media", fontsize=10)
    axes[1].set_ylim(0.0, 1.0)
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")
