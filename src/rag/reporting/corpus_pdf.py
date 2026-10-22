import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from rag.config import settings

def markdown_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0D3B66"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=3
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1D6996"),
        spaceBefore=10,
        spaceAfter=4
    )
    
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#333333"),
        spaceBefore=6,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=2
    )

    story = []
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CCCCCC"), spaceAfter=5, spaceBefore=5))
            continue
            
        # Clean formatting for reportlab
        line_clean = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Handle bold markdown **text**
        parts = line_clean.split("**")
        if len(parts) > 1:
            rebuilt = ""
            for i, p in enumerate(parts):
                if i % 2 == 1:
                    rebuilt += f"<b>{p}</b>"
                else:
                    rebuilt += p
            line_clean = rebuilt

        if line.startswith("# "):
            clean_text = line_clean.lstrip("# ").strip()
            story.append(Paragraph(clean_text, title_style))
        elif line.startswith("## "):
            clean_text = line_clean.lstrip("# ").strip()
            story.append(Paragraph(clean_text, h2_style))
        elif line.startswith("### "):
            clean_text = line_clean.lstrip("# ").strip()
            story.append(Paragraph(clean_text, h3_style))
        elif line.startswith("- ") or line.startswith("* "):
            clean_text = "• " + line_clean[2:].strip()
            story.append(Paragraph(clean_text, bullet_style))
        elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. ") or line.startswith("5. "):
            story.append(Paragraph(line_clean, bullet_style))
        else:
            if "Ministerio de Salud" in line or "Colegio Americano" in line or "Sociedad Europea" in line or "Código Documental" in line:
                story.append(Paragraph(line_clean, subtitle_style))
            else:
                story.append(Paragraph(line_clean, body_style))

    doc.build(story)
    print(f"Generated PDF: {pdf_path}")


def generate_corpus_pdfs(documents_dir: str | None = None) -> list[str]:
    """Convierte cada documento Markdown del corpus en su versión PDF."""
    documents_dir = documents_dir or str(settings.documents_dir)
    generated = []
    for fname in sorted(os.listdir(documents_dir)):
        if not fname.endswith(".md"):
            continue
        pdf_name = fname.replace(".md", ".pdf")
        pdf_path = os.path.join(documents_dir, pdf_name)
        markdown_to_pdf(os.path.join(documents_dir, fname), pdf_path)
        generated.append(pdf_path)
    return generated


if __name__ == "__main__":
    generate_corpus_pdfs()
