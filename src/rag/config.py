"""
Configuración centralizada del proyecto ACA N° 1 — Computación en la Nube.

Este módulo es la única fuente de verdad para:
  * Credenciales del clúster vectorial (Qdrant Cloud) — leídas **solo** del `.env`.
  * Modelo de embeddings y sus hiperparámetros.
  * Parámetros de fragmentación (chunking) e indexación por lotes.
  * Rutas del proyecto (datos, salida, documentos) resueltas de forma absoluta,
    de modo que los scripts funcionen sin importar desde dónde se invoquen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client.http.models import Distance

# Raíz del repositorio: src/rag/config.py -> src/rag -> src -> raíz
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carga del .env (si existe) antes de leer cualquier variable de entorno.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Configuración inmutable del pipeline RAG."""

    # --- Credenciales del almacén vectorial cloud (nunca hardcodeadas) ---
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    collection_name: str = os.getenv(
        "QDRANT_COLLECTION_NAME", "clinical_heart_disease_rag"
    ).strip()

    # --- Modelo de embeddings ---
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    vector_dimension: int = 384
    distance_metric: Distance = Distance.COSINE

    # --- Estrategia de fragmentación ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Indexación por lotes ---
    batch_size: int = 64

    # --- Rutas del proyecto ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    output_dir: Path = PROJECT_ROOT / "output"
    notebook_path: Path = PROJECT_ROOT / "notebooks" / "heart_disease_rag_qdrant.ipynb"

    @property
    def is_cloud_mode(self) -> bool:
        """True si hay credenciales para operar contra el clúster cloud."""
        return bool(self.qdrant_url and self.qdrant_api_key)

    @property
    def benchmark_results_path(self) -> Path:
        """Ruta del JSON con los resultados medidos del benchmark semántico."""
        return self.output_dir / "benchmark_summary.json"

    @property
    def benchmark_csv_path(self) -> Path:
        """Ruta del CSV con el detalle Top-k de cada consulta del benchmark."""
        return self.output_dir / "benchmark_results.csv"


settings = Settings()
