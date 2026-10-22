"""
Interfaz de línea de comandos del proyecto ACA N° 1.

Un único comando (`rag`) con subcomandos claros, pensados para que cualquier
persona del equipo pueda reproducir cualquier entregable en un comando:

    rag pipeline     → ejecuta el pipeline completo (PASO 1 a 5) y guarda métricas
    rag evaluate     → solo el benchmark semántico y sus métricas
    rag ask "..."    → una consulta semántica puntual
    rag chat         → consola interactiva de consultas
    rag poster       → regenera el póster infografía en PDF
    rag diagram      → regenera el diagrama de arquitectura cloud
    rag corpus-pdf   → regenera los PDF del corpus documental
    rag summary-pdf  → regenera la guía fácil de resultados en PDF
    rag notebook     → reejecuta el notebook y guarda sus outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from qdrant_client.http import models

from rag.config import settings
from rag.evaluation import BENCHMARK_QUERIES, run_retrieval_benchmark, summarize_results
from rag.query_parser import UNSET, describe_filter_conditions, parse_query
from rag.vector_store import build_metadata_filter, get_qdrant_client, semantic_search

logger = logging.getLogger(__name__)

FILTER_CHOICES = {
    "all": None,
    "guidelines": "clinical_guideline",
    "diagnostic": "diagnostic_protocol",
    "pharmacology": "pharmacology_protocol",
    "patients": "patient_record",
}

FILTER_LABELS = {
    "all": "Sin filtro (cross-modal)",
    "guidelines": "Guías de práctica clínica",
    "diagnostic": "Protocolos diagnósticos",
    "pharmacology": "Protocolos farmacológicos",
    "patients": "Historias clínicas de pacientes",
}


def _build_filter(
    filter_type: str = "all",
    *,
    sex: str | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    disease: str | None = None,
    risk: str | None = None,
) -> models.Filter | None:
    """Construye el filtro de metadatos combinando las condiciones solicitadas."""
    doc_type = FILTER_CHOICES.get(filter_type)
    disease_flag: bool | None = None
    if disease in {"true", "sí", "si", "confirmada", "1"}:
        disease_flag = True
    elif disease in {"false", "no", "0"}:
        disease_flag = False

    return build_metadata_filter(
        doc_type,
        sex=sex,
        min_age=min_age,
        max_age=max_age,
        disease=disease_flag,
        risk=risk,
    )


# --------------------------------------------------------------------------- #
# Subcomandos
# --------------------------------------------------------------------------- #


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Ejecuta el pipeline completo de extremo a extremo."""
    from rag.pipeline import run_pipeline

    result = run_pipeline(reindex=not args.no_reindex, evaluate=not args.no_evaluate, top_k=args.top_k)
    print("\n" + "=" * 78)
    print("PIPELINE COMPLETADO")
    print("=" * 78)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nMétricas guardadas en: {settings.benchmark_results_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Ejecuta el benchmark semántico sobre la colección ya indexada."""
    from rag.pipeline import load_embedding_model, measure_latency, describe_collection
    from rag.evaluation import plot_similarity_metrics, save_summary

    client = get_qdrant_client()
    model = load_embedding_model()
    df = run_retrieval_benchmark(client, model, top_k=args.top_k)
    summary = summarize_results(df)
    summary["latency"] = measure_latency(client, model, top_k=args.top_k)
    summary["corpus"] = describe_collection(client)
    summary["collection_name"] = settings.collection_name
    summary["mode"] = "cloud" if settings.is_cloud_mode else "local"

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.benchmark_csv_path, index=False)
    save_summary(summary)
    plot_similarity_metrics(df, str(settings.output_dir / "similarity_distribution.png"))

    print("\n" + "=" * 78)
    print("BENCHMARK DE BÚSQUEDA SEMÁNTICA")
    print("=" * 78)
    print(f"{'ID':<5}{'Top-1':<10}{'Filtro':<52}Fuente")
    for q in summary["queries"]:
        print(f"{q['id']:<5}{q['top1_score']:<10.4f}{q['filter_label'][:50]:<52}{q['source_title'][:40]}")
    print(f"\nSimilitud media Top-1: {summary['top1_mean']:.4f} | Máximo: {summary['top1_max']:.4f}")
    latency = summary.get("latency", {})
    if latency:
        print(
            f"Latencia de consulta remota: mediana {latency.get('search_median_ms', 0):.0f} ms "
            f"(media {latency['search_mean_ms']:.0f} ms, mín {latency['search_min_ms']:.0f} / "
            f"máx {latency['search_max_ms']:.0f} ms) | vectorización local: {latency['embedding_ms']:.0f} ms"
        )
    return 0


def _fila_compacta(rank: int, res: dict) -> str:
    """Fila resumida de un resultado: score, paciente, edad, sexo, enfermedad y riesgo."""
    meta = res["metadata"]
    if meta.get("patient_id") is not None:
        fuente = f"Paciente #{meta['patient_id']}"
        extra = (
            f"{meta.get('age', '?')} años · {meta.get('sex', '?')} · "
            f"{'enfermedad' if meta.get('disease_present') else 'sin enfermedad'} · "
            f"{meta.get('risk_level', '?')}"
        )
    else:
        fuente = meta.get("title", "documento")
        extra = meta.get("doc_type", "?")
    return f"  {f'[{rank}]':<6}{res['score']:>7.4f}  │  {fuente:<40}│  {extra}"


def cmd_ask(args: argparse.Namespace) -> int:
    """Ejecuta una consulta semántica puntual."""
    from rag.pipeline import load_embedding_model

    client = get_qdrant_client()
    model = load_embedding_model()

    # Interpretación automática de la consulta: un embedding no puede comparar
    # números, así que las restricciones estructuradas se extraen del texto y
    # se aplican como filtro de payload.
    parseado = parse_query(args.query)
    usar_auto = not args.no_auto_filter

    # Los filtros manuales tienen PRIORIDAD sobre los detectados.
    manual_doc_type = FILTER_CHOICES.get(args.filter) if args.filter != "all" else UNSET
    manual_sex = args.sex if args.sex else UNSET
    manual_min_age = args.min_age if args.min_age is not None else UNSET
    manual_max_age = args.max_age if args.max_age is not None else UNSET
    manual_disease: Any = UNSET
    if args.disease in {"true", "sí", "si", "confirmada", "1"}:
        manual_disease = True
    elif args.disease in {"false", "no", "0"}:
        manual_disease = False
    manual_risk = args.risk if args.risk else UNSET

    if usar_auto:
        query_filter = parseado.to_filter(
            doc_type=manual_doc_type,
            sex=manual_sex,
            min_age=manual_min_age,
            max_age=manual_max_age,
            disease=manual_disease,
            risk=manual_risk,
        )
    else:
        query_filter = _build_filter(
            args.filter,
            sex=args.sex,
            min_age=args.min_age,
            max_age=args.max_age,
            disease=args.disease,
            risk=args.risk,
        )

    condiciones_finales = describe_filter_conditions(query_filter) if query_filter else "sin filtro"

    t0 = time.perf_counter()
    results = semantic_search(
        client=client,
        model=model,
        query=args.query,
        collection_name=settings.collection_name,
        limit=args.top_k,
        query_filter=query_filter,
    )
    args.elapsed_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 78)
    print(f"CONSULTA: {args.query}")
    if usar_auto and parseado.conditions:
        print(f"🧠 INTERPRETÉ LA CONSULTA: {' · '.join(parseado.conditions)}")
    elif usar_auto:
        print("🧠 INTERPRETÉ LA CONSULTA: sin restricciones detectables")
    else:
        print("🧠 Interpretación automática: DESACTIVADA (--no-auto-filter)")
    print(f"FILTRO APLICADO: {condiciones_finales}")
    print("=" * 78)
    if not results:
        print("No se encontraron resultados con ese filtro.")
        return 1

    if args.brief:
        # Modo compacto: ideal para grabar en video.
        print(f"\n  {'RANK':<6}{'SCORE':>7}  │  {'FUENTE':<40}│  DETALLE")
        for rank, res in enumerate(results, start=1):
            print(_fila_compacta(rank, res))
        print(f"\n  {len(results)} resultados en {args.elapsed_ms:.0f} ms\n")
        return 0

    for rank, res in enumerate(results, start=1):
        meta = res["metadata"]
        source = meta.get("title") or f"Paciente #{meta.get('patient_id', 'N/A')}"
        print(f"\n[{rank}] similitud coseno = {res['score']:.4f} | tipo = {meta.get('doc_type', 'N/A')}")
        print(f"    fuente : {source}")
        if meta.get("patient_id"):
            print(
                f"    paciente: {meta.get('age')} años | sexo: {meta.get('sex')} | "
                f"riesgo: {meta.get('risk_level')}"
            )
        for line in res["text"].splitlines():
            if line.strip():
                print(f"    | {line.strip()}")
    print()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Consola interactiva de consultas semánticas."""
    print("\n🩺 ASISTENTE RAG CLOUD DE SALUD CARDIOVASCULAR (Qdrant Cloud)")
    print("Escriba su consulta en lenguaje natural. 'salir' para terminar.\n")
    while True:
        try:
            user_input = input("❓ Consulta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta luego!")
            return 0
        if user_input.lower() in {"salir", "exit", "quit", "q"}:
            print("¡Hasta luego!")
            return 0
        if not user_input:
            continue
        cmd_ask(
            argparse.Namespace(
                query=user_input,
                top_k=args.top_k,
                filter=args.filter,
                sex=args.sex,
                min_age=args.min_age,
                max_age=args.max_age,
                disease=args.disease,
                risk=args.risk,
            )
        )


def cmd_poster(args: argparse.Namespace) -> int:
    """Regenera el póster tipo infografía."""
    from rag.reporting.poster import generate_poster_pdf

    generate_poster_pdf(args.output)
    return 0


def cmd_diagram(args: argparse.Namespace) -> int:
    """Regenera el diagrama de arquitectura cloud."""
    from rag.reporting.architecture import generate_architecture_diagram

    generate_architecture_diagram(args.output)
    return 0


def cmd_corpus_pdf(args: argparse.Namespace) -> int:
    """Regenera los PDF del corpus documental a partir de los Markdown."""
    from rag.reporting.corpus_pdf import generate_corpus_pdfs

    generate_corpus_pdfs()
    return 0


def cmd_summary_pdf(args: argparse.Namespace) -> int:
    """Regenera la guía fácil de resultados."""
    from rag.reporting.summary_pdf import generate_easy_pdf

    generate_easy_pdf(args.output)
    return 0


def cmd_notebook_report(args: argparse.Namespace) -> int:
    """Genera el informe del notebook en PDF y HTML."""
    from rag.reporting.notebook_report import generate_notebook_report

    pdf, html = generate_notebook_report()
    print(f"✅ Informe PDF : {pdf}")
    print(f"✅ Informe HTML: {html}")
    return 0


def cmd_notebook(args: argparse.Namespace) -> int:
    """Reejecuta el notebook reproducible y guarda sus outputs."""
    from scripts.execute_notebook import run_notebook  # type: ignore[import-not-found]

    run_notebook(str(settings.notebook_path))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenera la documentación de resultados en Markdown."""
    from rag.reporting.results_doc import write_results_markdown

    target = write_results_markdown()
    print(f"✅ Documento de resultados generado en: {target}")
    return 0


def cmd_queries(args: argparse.Namespace) -> int:
    """Muestra la batería oficial de consultas y sus filtros."""
    print("\nBATERÍA OFICIAL DE CONSULTAS DE VALIDACIÓN\n")
    for bq in BENCHMARK_QUERIES:
        print(f"{bq['id']} — {bq['description']}")
        print(f"    pregunta: {bq['query']}")
        print(f"    filtro  : {bq['filter_label']}")
        print(f"    esperado: {bq['expected']}\n")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Sistema RAG cloud serverless en salud — ACA N° 1 Computación en la Nube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Muestra logs detallados (INFO).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", help="Ejecuta el pipeline completo (PASO 1 a 5).")
    p.add_argument("--top-k", type=int, default=5, help="Fragmentos recuperados por consulta.")
    p.add_argument("--no-reindex", action="store_true", help="No recrea ni reindexa la colección.")
    p.add_argument("--no-evaluate", action="store_true", help="No ejecuta el benchmark final.")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("evaluate", help="Ejecuta solo el benchmark semántico.")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("ask", help="Consulta semántica puntual.")
    p.add_argument("query", help="Pregunta en lenguaje natural.")
    p.add_argument("--top-k", "-k", type=int, default=3)
    p.add_argument(
        "--filter",
        "-f",
        choices=list(FILTER_CHOICES),
        default="all",
        help="Filtro por tipo de documento.",
    )
    p.add_argument("--sex", default=None, help="Filtro por sexo (Masculino / Femenino).")
    p.add_argument("--min-age", type=int, default=None, help="Edad mínima (inclusive).")
    p.add_argument("--max-age", type=int, default=None, help="Edad máxima (inclusive).")
    p.add_argument(
        "--disease",
        default=None,
        help="Filtro por enfermedad coronaria confirmada: true / false.",
    )
    p.add_argument("--risk", default=None, help="Filtro por nivel de riesgo (Bajo, Alto, ...).")
    p.add_argument(
        "--brief",
        "-b",
        action="store_true",
        help="Salida compacta en tabla (ideal para demos en video).",
    )
    p.add_argument(
        "--no-auto-filter",
        action="store_true",
        help="Desactiva la interpretación automática de la consulta (no se extraen filtros del texto).",
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Consola interactiva de consultas.")
    p.add_argument("--top-k", "-k", type=int, default=3)
    p.add_argument("--filter", "-f", choices=list(FILTER_CHOICES), default="all")
    p.add_argument("--sex", default=None)
    p.add_argument("--min-age", type=int, default=None)
    p.add_argument("--max-age", type=int, default=None)
    p.add_argument("--disease", default=None)
    p.add_argument("--risk", default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("queries", help="Muestra la batería oficial de consultas.")
    p.set_defaults(func=cmd_queries)

    p = sub.add_parser("poster", help="Regenera el póster infografía en PDF.")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_poster)

    p = sub.add_parser("diagram", help="Regenera el diagrama de arquitectura.")
    p.add_argument("--output", default=str(settings.output_dir / "cloud_rag_architecture.png"))
    p.set_defaults(func=cmd_diagram)

    p = sub.add_parser("corpus-pdf", help="Regenera los PDF del corpus documental.")
    p.set_defaults(func=cmd_corpus_pdf)

    p = sub.add_parser("summary-pdf", help="Regenera la guía fácil de resultados.")
    p.add_argument("--output", default=str(settings.output_dir / "Guia_Facil_Resultados_RAG.pdf"))
    p.set_defaults(func=cmd_summary_pdf)

    p = sub.add_parser("report", help="Regenera docs/RESULTADOS.md con las métricas medidas.")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("notebook-report", help="Genera el informe del notebook en PDF y HTML.")
    p.set_defaults(func=cmd_notebook_report)

    p = sub.add_parser("notebook", help="Reejecuta el notebook reproducible.")
    p.set_defaults(func=cmd_notebook)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
