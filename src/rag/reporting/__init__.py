"""
Generadores de los entregables en PDF/PNG del proyecto ACA N° 1.

  * ``poster``       → póster tipo infografía (entregable oficial 2).
  * ``architecture`` → diagrama de arquitectura cloud en PNG.
  * ``summary_pdf`` → guía fácil de interpretación de resultados.
  * ``corpus_pdf``  → conversión del corpus documental Markdown → PDF.
"""

from rag.reporting.architecture import generate_architecture_diagram
from rag.reporting.corpus_pdf import generate_corpus_pdfs, markdown_to_pdf
from rag.reporting.poster import generate_poster_pdf
from rag.reporting.results_doc import build_markdown, write_results_markdown
from rag.reporting.summary_pdf import generate_easy_pdf

__all__ = [
    "build_markdown",
    "generate_architecture_diagram",
    "generate_corpus_pdfs",
    "generate_easy_pdf",
    "generate_poster_pdf",
    "markdown_to_pdf",
    "write_results_markdown",
]
