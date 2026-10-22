"""
Genera el informe del notebook reproducible en PDF y HTML.

El informe se construye **leyendo el notebook ya ejecutado**, de modo que el
documento entregado nunca queda desactualizado respecto al cuaderno.

    uv run rag notebook-report
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from rag.config import settings

C_NAVY = colors.HexColor("#0F2942")
C_TEAL = colors.HexColor("#226B80")
C_BORDER = colors.HexColor("#D1E3EC")
C_CODE_BG = colors.HexColor("#F4F8FA")
C_MUTED = colors.HexColor("#475569")


# --------------------------------------------------------------------------- #
# Extracción de contenido del notebook
# --------------------------------------------------------------------------- #


def _read_notebook(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _text_outputs(cell: dict[str, Any]) -> str:
    """Concatena las salidas de texto de una celda de código."""
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            chunks.append("".join(text) if isinstance(text, list) else text)
        elif output.get("output_type") == "execute_result":
            data = output.get("data", {})
            if "text/plain" in data:
                text = data["text/plain"]
                chunks.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _images(cell: dict[str, Any]) -> list[bytes]:
    """Extrae las imágenes PNG generadas por una celda de código."""
    images: list[bytes] = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            try:
                images.append(base64.b64decode(data["image/png"]))
            except Exception:  # pragma: no cover
                continue
    return images


def _markdown_to_text(source: str) -> str:
    """Convierte el Markdown de una celda en texto plano legible."""
    texto = re.sub(r"^#{1,6}\s*", "", source, flags=re.MULTILINE)
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)
    texto = re.sub(r"`(.+?)`", r"\1", texto)
    texto = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", texto)
    texto = re.sub(r"^\s*[-*]\s+", "• ", texto, flags=re.MULTILINE)
    return texto.strip()


# --------------------------------------------------------------------------- #
# Informe en PDF
# --------------------------------------------------------------------------- #


def build_pdf(notebook_path: Path, output_path: Path) -> Path:
    """Construye el PDF del informe a partir del notebook ejecutado."""
    nb = _read_notebook(notebook_path)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX", parent=styles["Title"], fontSize=16, leading=20, textColor=C_NAVY, alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        "SubX", parent=styles["Normal"], fontSize=9, leading=12, textColor=C_MUTED
    )
    heading_style = ParagraphStyle(
        "HeadX", parent=styles["Heading2"], fontSize=12, leading=15, textColor=C_NAVY, spaceBefore=10
    )
    body_style = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=9, leading=12.5)
    code_style = ParagraphStyle(
        "CodeX",
        parent=styles["Code"],
        fontSize=7.4,
        leading=9.4,
        textColor=colors.HexColor("#0F172A"),
        backColor=C_CODE_BG,
        borderPadding=5,
    )
    out_style = ParagraphStyle(
        "OutX",
        parent=styles["Code"],
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#334155"),
        backColor=colors.HexColor("#FAFBFC"),
        borderPadding=5,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Resumen de Resultados del Notebook — ACA N° 1",
        author="Jaime Cabrera · CUN",
    )

    story: list[Any] = []
    temporales: list[Path] = []
    story.append(Paragraph("CUN · Especialización en Inteligencia Artificial · Computación en la Nube", subtitle_style))
    story.append(Paragraph("Informe del Notebook Reproducible — ACA N° 1", title_style))
    story.append(
        Paragraph(
            "Proyecto: Sistema RAG cloud serverless en salud sobre Qdrant Cloud con embeddings abiertos.<br/>"
            f"Cuaderno de origen: <b>{notebook_path.name}</b> · Colección: <b>{settings.collection_name}</b> · "
            f"Modelo: <b>{settings.embedding_model_name}</b>",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=C_TEAL, spaceBefore=6, spaceAfter=10))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = _cell_source(cell).strip()
            if not source:
                continue
            primera = source.splitlines()[0]
            if primera.startswith("#") and not primera.startswith("##"):
                story.append(Paragraph(_markdown_to_text(source[:1200]), heading_style))
            else:
                story.append(Paragraph(html.escape(_markdown_to_text(source))[:2600].replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 4))
        else:
            source = _cell_source(cell).strip()
            if source:
                tabla = Table(
                    [[Paragraph(html.escape(source).replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style)]],
                    colWidths=[178 * mm],
                )
                tabla.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), C_CODE_BG),
                            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(tabla)
                story.append(Spacer(1, 3))

            salida = _text_outputs(cell)
            if salida:
                tabla_out = Table(
                    [[Paragraph(html.escape(salida)[:3500].replace("\n", "<br/>"), out_style)]],
                    colWidths=[178 * mm],
                )
                tabla_out.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFC")),
                            ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(tabla_out)
                story.append(Spacer(1, 3))

            for index, imagen in enumerate(_images(cell)):
                ruta_tmp = output_path.parent / f"_tmp_notebook_fig_{len(temporales)}_{index}.png"
                ruta_tmp.write_bytes(imagen)
                temporales.append(ruta_tmp)
                story.append(Image(str(ruta_tmp), width=165 * mm, height=68 * mm))
                story.append(Spacer(1, 6))

    doc.build(story)
    for ruta in temporales:
        ruta.unlink(missing_ok=True)
    return output_path


# --------------------------------------------------------------------------- #
# Informe en HTML
# --------------------------------------------------------------------------- #


def build_html(notebook_path: Path, output_path: Path) -> Path:
    """Construye el informe HTML del notebook (ligero, sin CDN)."""
    nb = _read_notebook(notebook_path)
    # El informe comparte el sistema de diseño de la app (metáfora de
    # expediente clínico): papel, tinta, azul de escribir y mono para los
    # datos. Ver app/theme.py.
    partes: list[str] = [
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
        "<title>Informe del Notebook — ACA N° 1</title>",
        "<link rel='preconnect' href='https://fonts.googleapis.com'>",
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>",
        "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'"
        "&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;600;700'"
        "&display=swap' rel='stylesheet'>",
        "<style>"
        ":root{--papel:#E9EAE6;--hoja:#FBFBF9;--tinta:#1C1F1D;--tinta-2:#5A615C;"
        "--tinta-3:#8A918B;--regla:#C4C9C2;--regla-suave:#DCDED8;--azul:#22485E;--sello:#8C3A2E;"
        "--sombra:3px 3px 0 rgba(28,31,29,.09)}"
        "body{font-family:'IBM Plex Sans',system-ui,sans-serif;max-width:900px;margin:2.5rem auto;"
        "padding:0 1.4rem;color:var(--tinta);background-color:var(--papel);"
        "background-image:repeating-linear-gradient(to bottom,transparent 0 29px,rgba(28,31,29,.028) 29px 30px);"
        "line-height:1.65}"
        "h1,h2{font-family:'IBM Plex Serif',Georgia,serif;color:var(--tinta);letter-spacing:-.012em}"
        "h1{font-size:1.9rem;margin:0 0 .6rem;padding-top:1rem;border-top:6px solid var(--azul);"
        "display:inline-block}"
        "h2{font-size:1.25rem;margin:2rem 0 .6rem;padding-bottom:.4rem;border-bottom:1px solid var(--regla)}"
        "p{color:var(--tinta-2)}"
        "b{color:var(--tinta);font-weight:600}"
        "code{font-family:'IBM Plex Mono',monospace;background:rgba(34,72,94,.08);border:1px solid rgba(34,72,94,.14);"
        "border-radius:2px;padding:.05em .3em;font-size:.86em;color:var(--azul)}"
        "pre{font-family:'IBM Plex Mono',monospace;background:var(--hoja);border:1px solid var(--regla);"
        "box-shadow:var(--sombra);padding:.9rem 1rem;overflow-x:auto;font-size:.8rem;border-radius:2px;line-height:1.5}"
        "pre code{background:none;border:none;padding:0;color:var(--tinta-2)}"
        ".out{font-family:'IBM Plex Mono',monospace;background:var(--hoja);border:1px solid var(--regla);"
        "border-left:3px solid var(--azul);box-shadow:var(--sombra);padding:.7rem .9rem;white-space:pre-wrap;"
        "font-size:.79rem;line-height:1.55;color:var(--tinta-2)}"
        "img{max-width:100%;border:1px solid var(--regla);box-shadow:var(--sombra);background:var(--hoja)}"
        "</style></head><body>",
        "<h1>Informe del Notebook Reproducible — ACA N° 1</h1>",
        f"<p><b>Cuaderno:</b> {notebook_path.name} · <b>Colección:</b> {settings.collection_name} · "
        f"<b>Modelo:</b> {settings.embedding_model_name}</p>",
    ]
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            partes.append(f"<div>{_markdown_to_text(_cell_source(cell))}</div>")
        else:
            source = _cell_source(cell).strip()
            if source:
                partes.append(f"<pre><code>{html.escape(source)}</code></pre>")
            salida = _text_outputs(cell)
            if salida:
                partes.append(f"<div class='out'>{html.escape(salida)}</div>")
            for imagen in _images(cell):
                partes.append(f"<img src='data:image/png;base64,{base64.b64encode(imagen).decode()}'/>")
    partes.append("</body></html>")

    output_path.write_text("\n".join(partes), encoding="utf-8")
    return output_path


def generate_notebook_report(
    notebook_path: Path | None = None,
    pdf_path: Path | None = None,
    html_path: Path | None = None,
) -> tuple[Path, Path]:
    """Genera el informe del notebook en PDF y HTML."""
    notebook_path = notebook_path or settings.notebook_path
    pdf_path = pdf_path or settings.output_dir / "Resumen_Resultados_Notebook.pdf"
    html_path = html_path or settings.output_dir / "notebook_report.html"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    return build_pdf(notebook_path, pdf_path), build_html(notebook_path, html_path)
