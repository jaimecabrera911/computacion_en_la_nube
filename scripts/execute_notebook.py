"""
Reejecuta el notebook reproducible del proyecto y guarda sus outputs.

Uso:
    uv run python scripts/execute_notebook.py
    uv run rag notebook
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import settings  # noqa: E402


def run_notebook(nb_path: str | None = None) -> None:
    """Ejecuta el notebook de principio a fin y lo guarda con sus outputs."""
    path = Path(nb_path) if nb_path else settings.notebook_path
    print(f"Iniciando ejecución secuencial de {path}...")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Ejecución completa guardada en: {path}")


if __name__ == "__main__":
    run_notebook()
