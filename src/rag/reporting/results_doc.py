"""
Genera ``docs/RESULTADOS.md`` a partir del JSON de métricas medido.

De este modo la documentación de resultados nunca queda desactualizada respecto
de la última ejecución real del pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import settings
from rag.evaluation import load_summary


def build_markdown(summary: dict[str, Any]) -> str:
    """Construye el documento de resultados en Markdown a partir del resumen."""
    corpus = summary.get("corpus", {})
    latency = summary.get("latency", {})
    by_rank = summary.get("by_rank", {})
    stats = summary.get("per_query_stats", {})

    lines: list[str] = []
    lines.append("# 📊 Resultados Medidos del Benchmark Semántico")
    lines.append("")
    lines.append(
        "> **Este documento se genera automáticamente** con `uv run rag report`, a partir de "
        "`output/benchmark_summary.json` (el JSON que produce el propio pipeline al ejecutarse). "
        "Por lo tanto, **todos los números aquí corresponden a una ejecución real y medible** "
        "contra el clúster de Qdrant Cloud."
    )
    lines.append("")
    lines.append(
        f"**Colección:** `{summary.get('collection_name', settings.collection_name)}` · "
        f"**Modo:** {summary.get('mode', 'n/d')} · "
        f"**Modelo:** `{settings.embedding_model_name}`"
    )
    lines.append("")

    # --- Corpus ---
    lines.append("## 1. Corpus indexado")
    lines.append("")
    lines.append("| Componente | Cantidad |")
    lines.append("|---|---|")
    lines.append(f"| Viñetas clínicas (dataset UCI Heart Disease, 303 pacientes) | {corpus.get('patients', 'n/d')} |")
    lines.append(f"| Fragmentos documentales (3 guías clínicas) | {corpus.get('document_chunks', 'n/d')} |")
    lines.append(f"| **Total de vectores en la colección** | **{corpus.get('total_vectors', 'n/d')}** |")
    lines.append("")

    # --- Benchmark ---
    lines.append("## 2. Batería oficial de consultas (Top-1 y Top-5)")
    lines.append("")
    lines.append("| ID | Enfoque | Filtro Qdrant | Top-1 | Media Top-5 | Mín | Máx | Fuente recuperada |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for q in summary.get("queries", []):
        st = stats.get(q["id"], {})
        lines.append(
            f"| **{q['id']}** | {q['description']} | `{q['filter_label']}` | "
            f"**{q['top1_score']:.4f}** | {st.get('mean', 'n/d')} | {st.get('min', 'n/d')} | "
            f"{st.get('max', 'n/d')} | {q['source_title']} |"
        )
    lines.append("")

    # --- Agregados ---
    lines.append("## 3. Métricas agregadas")
    lines.append("")
    lines.append("| Métrica | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Similitud coseno media (Top-1) | **{summary.get('top1_mean', 'n/d')}** |")
    lines.append(f"| Similitud coseno máxima (Top-1) | {summary.get('top1_max', 'n/d')} |")
    lines.append(f"| Similitud coseno mínima (Top-1) | {summary.get('top1_min', 'n/d')} |")
    lines.append(f"| Similitud coseno media (todos los Top-5) | {summary.get('all_mean', 'n/d')} |")
    if latency:
        lines.append(f"| Latencia de búsqueda remota (mediana) | **{latency.get('search_median_ms', latency.get('search_mean_ms', 'n/d'))} ms** |")
        lines.append(f"| Latencia de búsqueda remota (media) | {latency.get('search_mean_ms', 'n/d')} ms |")
        lines.append(f"| Latencia mínima / máxima | {latency.get('search_min_ms', 'n/d')} / {latency.get('search_max_ms', 'n/d')} ms |")
        lines.append(f"| Vectorización local (SBERT en CPU) | {latency.get('embedding_ms', 'n/d')} ms |")
    if summary.get("embedding_seconds"):
        lines.append(f"| Tiempo de indexación de los {corpus.get('total_vectors', '?')} vectores | {summary['embedding_seconds']} s |")
    lines.append("")

    # --- Degradación por rango ---
    if by_rank:
        lines.append("## 4. Degradación de la similitud por rango de recuperación")
        lines.append("")
        lines.append(
            "Una búsqueda semántica sana **ordena por relevancia**: el score del Rank 1 debe ser el "
            "mayor y debe decrecer de forma monótona. Si no fuera así, el sistema no estaría "
            "ordenando por relevancia."
        )
        lines.append("")
        lines.append("| Rank | Similitud media |")
        lines.append("|---|---|")
        for rank in sorted(by_rank, key=lambda r: int(r)):
            lines.append(f"| {rank} | {by_rank[rank]:.4f} |")
        lines.append("")
        monotona = all(
            by_rank[k] >= by_rank[str(int(k) + 1)] for k in ["1", "2", "3", "4"] if str(int(k) + 1) in by_rank
        )
        lines.append(f"**Degradación monótona decreciente:** {'sí ✅' if monotona else 'no ⚠️'}")
        lines.append("")

    # --- Fragmentos recuperados ---
    lines.append("## 5. Fragmentos recuperados (Top-1 de cada consulta)")
    lines.append("")
    for q in summary.get("queries", []):
        lines.append(f"### {q['id']} — {q['description']}")
        lines.append("")
        lines.append(f"- **Pregunta:** {q['query']}")
        lines.append(f"- **Filtro:** `{q['filter_label']}`")
        lines.append(f"- **Similitud coseno:** `{q['top1_score']:.4f}`")
        lines.append(f"- **Fuente recuperada:** {q['source_title']}")
        lines.append(f"- **Resultado esperado:** {q['expected']}")
        snippet = (q.get("snippet") or "").strip()
        if snippet:
            lines.append("- **Fragmento devuelto:**")
            lines.append("")
            lines.append("  > " + snippet.replace("\n", " ").replace("|", "\\|")[:400])
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Documento generado automáticamente. Para regenerarlo: `uv run rag report`.*"
    )
    lines.append("")

    return "\n".join(lines)


def write_results_markdown(path: str | Path | None = None) -> Path:
    """Escribe ``docs/RESULTADOS.md`` a partir del resumen de métricas."""
    summary = load_summary()
    if summary is None:
        raise FileNotFoundError(
            "No existe output/benchmark_summary.json. Ejecute primero `uv run rag pipeline`."
        )
    target = Path(path) if path else settings.project_root / "docs" / "RESULTADOS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_markdown(summary), encoding="utf-8")
    return target
