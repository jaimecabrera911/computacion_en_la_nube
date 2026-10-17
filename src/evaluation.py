import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from src.vector_store import semantic_search
from src.config import settings

BENCHMARK_QUERIES = [
    {
        "id": "Q1_CLINICAL_GUIDELINE",
        "description": "Protocolo de evaluación y manejo ante dolor torácico tipo angina típica y sospecha de isquemia miocárdica",
        "query": "¿Cuáles son los criterios clínicos para diagnosticar angina típica y cuál es el algoritmo de estratificación de riesgo en dolor torácico?",
        "filter": models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="clinical_guideline"))]
        )
    },
    {
        "id": "Q2_ERGOMETRY_DIAGNOSTIC",
        "description": "Criterios ergométricos de alto riesgo por depresión de segmento ST y pendiente descendente",
        "query": "¿Qué significado clínico tiene una depresión del segmento ST mayor a 2 mm con pendiente plana o descendente durante la prueba de esfuerzo?",
        "filter": models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="diagnostic_protocol"))]
        )
    },
    {
        "id": "Q3_PHARMACOLOGY_SECONDARY",
        "description": "Metas de colesterol LDL y terapia antiplaquetaria dual en cardiopatía isquémica confirmada",
        "query": "¿Cuáles son las metas de c-LDL con estatinas de alta potencia y el esquema de doble antiagregación plaquetaria con aspirina y ticagrelor?",
        "filter": models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="pharmacology_protocol"))]
        )
    },
    {
        "id": "Q4_PATIENT_COHORT_HIGH_RISK",
        "description": "Recuperación de pacientes masculinos mayores de 60 años con angina típica y estenosis multivaso",
        "query": "Paciente masculino mayor de 60 años con angina típica, hipertensión estadio 2, fluoroscopía con vasos mayores comprometidos y alto riesgo cardiovascular",
        "filter": models.Filter(
            must=[
                models.FieldCondition(key="doc_type", match=models.MatchValue(value="patient_record")),
                models.FieldCondition(key="sex", match=models.MatchValue(value="Masculino")),
                models.FieldCondition(key="disease_present", match=models.MatchValue(value=True))
            ]
        )
    },
    {
        "id": "Q5_HYBRID_CROSS_MODAL",
        "description": "Consulta híbrida abierta: correlación clínica y terapéutica para defecto de perfusión reversible (SPECT)",
        "query": "Tratamiento de revascularización y pronóstico para defecto de perfusión reversible en gammagrafía SPECT con angina inducida por ejercicio",
        "filter": None  # Sin filtro: evalúa capacidad de recuperar guías y pacientes afines
    }
]

def run_retrieval_benchmark(
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = 5
) -> pd.DataFrame:
    """Ejecuta una batería de consultas clínicas estructuradas y recopila métricas cuantitativas."""
    records = []
    
    for bq in BENCHMARK_QUERIES:
        results = semantic_search(
            client=client,
            model=model,
            query=bq["query"],
            collection_name=settings.collection_name,
            limit=top_k,
            query_filter=bq["filter"]
        )
        
        for rank, res in enumerate(results, start=1):
            records.append({
                "query_id": bq["id"],
                "query_desc": bq["description"],
                "rank": rank,
                "score": res["score"],
                "doc_type": res["metadata"].get("doc_type", "unknown"),
                "source_title": res["metadata"].get("title", f"Paciente #{res['metadata'].get('patient_id', 'N/A')}"),
                "snippet": res["text"][:120] + "..."
            })
            
    df_results = pd.DataFrame(records)
    return df_results

def plot_similarity_metrics(df_results: pd.DataFrame, output_path: str = "output/similarity_distribution.png"):
    """Genera gráficos profesionales de distribución de similitud coseno para el reporte técnico."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Boxplot de similitud por consulta
    sns.boxplot(
        data=df_results,
        x="query_id",
        y="score",
        ax=axes[0],
        palette="Blues_d",
        hue="query_id",
        legend=False
    )
    axes[0].set_title("Distribución de Similitud Coseno por Consulta (Top-5)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Identificador de Consulta", fontsize=10)
    axes[0].set_ylabel("Puntuación de Similitud Coseno", fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0.0, 1.0)
    
    # Gráfico 2: Similitud media por rango (Rank 1 a 5)
    rank_means = df_results.groupby("rank")["score"].mean().reset_index()
    sns.barplot(
        data=rank_means,
        x="rank",
        y="score",
        ax=axes[1],
        palette="crest",
        hue="rank",
        legend=False
    )
    axes[1].set_title("Similitud Coseno Media según Posición de Recuperación (Rank 1-5)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Rango de Recuperación (Top-k)", fontsize=10)
    axes[1].set_ylabel("Similitud Coseno Media", fontsize=10)
    axes[1].set_ylim(0.0, 1.0)
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")
