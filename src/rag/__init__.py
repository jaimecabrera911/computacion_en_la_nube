"""
Sistema RAG cloud serverless para el proyecto ACA N° 1 — Computación en la Nube.

Pilar: Salud (cardiopatía isquémica). Corpus: dataset estructurado UCI Heart
Disease (ID 45) + corpus documental no estructurado (3 guías clínicas).

Paquetes:
  * ``rag.config``          → configuración y credenciales.
  * ``rag.dataset_loader``  → ingesta del dataset tabular y narrativización.
  * ``rag.document_processor`` → carga y fragmentación del corpus documental.
  * ``rag.vector_store``    → cliente Qdrant, indexación y búsqueda semántica.
  * ``rag.evaluation``      → benchmark de recuperación y métricas.
  * ``rag.pipeline``        → orquestador de extremo a extremo.
  * ``rag.cli``             → interfaz de línea de comandos.
  * ``rag.reporting``       → generación de póster, diagramas y guías PDF.
"""

__all__ = [
    "config",
    "dataset_loader",
    "document_processor",
    "vector_store",
    "evaluation",
    "pipeline",
    "cli",
]

__version__ = "1.0.0"
