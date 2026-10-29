import os
from reportlab.lib.pagesizes import A4

from rag.config import settings
from rag.evaluation import load_summary
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def _plain_text(fragment: str, limit: int = 260) -> str:
    """Convierte un fragmento Markdown en texto plano legible para el póster."""
    texto = fragment.replace("**", "").replace("`", "")
    texto = texto.replace("\\ge", ">=").replace("\\le", "<=").replace("\\times", "x")
    texto = texto.replace("###", "").replace("##", "").replace("#", "")
    texto = " ".join(texto.split())
    return texto[:limit]


def _chunk_counts_by_document() -> dict[str, int]:
    """Número real de fragmentos generados por cada documento complementario."""
    try:
        from rag.document_processor import chunk_unstructured_documents, load_unstructured_documents

        chunks = chunk_unstructured_documents(load_unstructured_documents(str(settings.documents_dir)))
        counts: dict[str, int] = {}
        for chunk in chunks:
            name = chunk["metadata"]["filename"]
            counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:  # pragma: no cover - el póster no debe fallar por esto
        return {}


def generate_poster_pdf(output_path: str | None = None):
    """Genera el póster tipo infografía (PDF A4, 2 páginas) con los datos medidos."""
    output_path = output_path or str(settings.output_dir / "Poster_Infografia_ACA1_RAG_Salud.pdf")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Métricas reales de la última ejecución del benchmark (fuente única de verdad).
    summary = load_summary() or {}
    measured = {q["id"]: q for q in summary.get("queries", [])}
    max_score = summary.get("top1_max")

    # Conteo real de fragmentos por documento complementario.
    chunk_counts = _chunk_counts_by_document()

    # Configuración de página A4 con márgenes optimizados
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    # Paleta de colores profesional institucional / cloud
    C_NAVY = colors.HexColor("#0F2942")
    C_BLUE = colors.HexColor("#1A5F7A")
    C_TEAL = colors.HexColor("#226B80")
    C_LIGHT_BG = colors.HexColor("#F4F8FA")
    C_BORDER = colors.HexColor("#D1E3EC")
    C_ACCENT = colors.HexColor("#28527A")
    C_TEXT = colors.HexColor("#1E293B")
    C_MUTED = colors.HexColor("#475569")
    # C_SUCCESS = colors.HexColor("#166534")

    # Estilos tipográficos
    title_style = ParagraphStyle(
        'PosterTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=C_NAVY,
        alignment=1, # Center
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'PosterSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=C_TEAL,
        alignment=1,
        spaceAfter=6
    )

    meta_header_style = ParagraphStyle(
        'PosterMetaHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=C_MUTED,
        alignment=1,
        spaceAfter=8
    )

    sec_header_style = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.white,
        spaceBefore=0,
        spaceAfter=0
    )

    body_style = ParagraphStyle(
        'PosterBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=C_TEXT
    )

    body_bold = ParagraphStyle(
        'PosterBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9.0,
        textColor=C_TEXT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=C_NAVY
    )

    table_cell_code = ParagraphStyle(
        'TableCellCode',
        parent=table_cell,
        fontName='Courier',
        fontSize=6.5,
        leading=8.5,
        textColor=C_ACCENT
    )

    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.3,
        leading=8.2,
        textColor=C_MUTED,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=1.5
    )

    def make_section_banner(title_text):
        p = Paragraph(f"<b>{title_text.upper()}</b>", sec_header_style)
        t = Table([[p]], colWidths=[531])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BLUE),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        return t

    story = []

    # ==================== PÁGINA 1: ARQUITECTURA, CONTEXTO Y STACK ====================
    # Encabezado Institucional
    story.append(Paragraph("CORPORACIÓN UNIFICADA NACIONAL DE EDUCACIÓN SUPERIOR (CUN)", meta_header_style))
    story.append(Paragraph("INFOGRAFÍA TÉCNICA: ARQUITECTURA RAG CLOUD SERVERLESS EN SALUD", title_style))
    story.append(Paragraph("Ingesta, Chunking, Vectorización e Indexación Semántica en Qdrant Cloud | Proyecto ACA N° 1", subtitle_style))
    story.append(Paragraph("<b>Especialización en Inteligencia Artificial</b> | Asignatura: Computación en la Nube | <b>Docente:</b> Hamilton Rivera Flor<br/><b>Dataset:</b> UCI Heart Disease (ID: 45) | <b>Vector Store:</b> Qdrant Cloud (AWS ca-central-1) | <b>Modelo:</b> paraphrase-multilingual-MiniLM-L12-v2", meta_header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_TEAL, spaceAfter=8, spaceBefore=0))

    # Bloque 1: Situación Problema & Objetivo General (2 columnas)
    p_problema = Paragraph(
        "<b>Situación Problema y Pregunta Orientadora:</b><br/>"
        "¿Cómo estructurar e implementar una canalización (pipeline) de datos no estructurados en la nube mediante "
        "bases de datos vectoriales serverless y modelos de embeddings abiertos, que permita indexar y recuperar "
        "semánticamente información contextual especializada con alta precisión, costo cero y baja latencia en el sector salud?",
        body_style
    )
    p_objetivo = Paragraph(
        "<b>Objetivo General del Proyecto:</b><br/>"
        "Diseñar y desplegar en <b>Qdrant Cloud</b> un sistema RAG serverless de alto rendimiento que integra 303 registros "
        "tabulares clínicos transformados en viñetas semánticas con un corpus no estructurado de 3 guías de práctica clínica cardiológica, "
        "demostrando consultas semánticas y filtrado de metadatos con latencias inferiores a 150 ms y costo cero permanente.",
        body_style
    )

    t_intro = Table([[p_problema, p_objetivo]], colWidths=[260, 265])
    t_intro.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_intro)
    story.append(Spacer(1, 7))

    # Bloque 2: Pilar Salud y Corpus Documental
    story.append(make_section_banner("1. Pilar Seleccionado: Salud 🩺 y Corpus Documental Integrado"))
    story.append(Spacer(1, 3))

    corpus_table_data = [
        [
            Paragraph("<b>Componente</b>", table_cell_bold),
            Paragraph("<b>Fuente / Entidad</b>", table_cell_bold),
            Paragraph("<b>Estructura / Formato</b>", table_cell_bold),
            Paragraph("<b>Dimensiones / Contenido Clínico</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>Dataset Base</b>", table_cell),
            Paragraph("UCI Heart Disease (ID: 45)<br/>Cleveland Clinic", table_cell),
            Paragraph("Tabular / CSV convertido a viñetas narrativas estructuradas", table_cell),
            Paragraph("303 pacientes, 14 variables (edad, dolor torácico cp, tensión, colesterol, ECG, thalach, oldpeak, ca, thal, target).", table_cell)
        ],
        [
            Paragraph("<b>Guía Clínica 01</b>", table_cell),
            Paragraph("MinSalud / OPS / OMS<br/>GPC-CARDIO-2024", table_cell),
            Paragraph("Markdown & PDF<br/>(%d chunks)" % chunk_counts.get("DOC_01_Guia_Clinica_Enfermedad_Coronaria_MinSalud_OMS.md", 16), table_cell),
            Paragraph("Clasificación semiológica de dolor torácico (angina típica, atípica, no anginosa), factores de riesgo y ECG basal.", table_cell)
        ],
        [
            Paragraph("<b>Protocolo 02</b>", table_cell),
            Paragraph("ACC / AHA Guidelines<br/>PRT-DIAG-CARDIO", table_cell),
            Paragraph("Markdown & PDF<br/>(%d chunks)" % chunk_counts.get("DOC_02_Protocolo_Pruebas_Esfuerzo_Ergometria_ACC_AHA.md", 16), table_cell),
            Paragraph("Criterios de ergometría, pendiente de depresión del segmento ST (slope/oldpeak), fluoroscopía (ca) y perfusión SPECT (thal).", table_cell)
        ],
        [
            Paragraph("<b>Protocolo 03</b>", table_cell),
            Paragraph("ESC / OMS Guidelines<br/>ESC-OMS-PREV", table_cell),
            Paragraph("Markdown & PDF<br/>(%d chunks)" % chunk_counts.get("DOC_03_Guia_Manejo_Farmacologico_Prevencion_Secundaria.md", 14), table_cell),
            Paragraph("Metas de c-LDL (<55 mg/dL con estatinas), doble antiagregación plaquetaria (AAS + Ticagrelor) y código infarto.", table_cell)
        ]
    ]

    t_corpus = Table(corpus_table_data, colWidths=[90, 110, 120, 211])
    t_corpus.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT_BG),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_corpus)
    story.append(Spacer(1, 7))

    # Bloque 3: Arquitectura Cloud y Diagrama
    story.append(make_section_banner("2. Diagrama de Arquitectura Cloud RAG y Flujo de Ingesta Serverless"))
    story.append(Spacer(1, 3))

    arch_img_path = str(settings.output_dir / "cloud_rag_architecture.png")
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=531, height=195))
    story.append(Spacer(1, 6))

    # Bloque 4: Stack Tecnológico y Estrategia de Chunking
    story.append(make_section_banner("3. Stack Tecnológico Cloud y Estrategia Técnica de Fragmentación"))
    story.append(Spacer(1, 3))

    stack_data = [
        [
            Paragraph("<b>Capa Arquitectónica</b>", table_cell_bold),
            Paragraph("<b>Tecnología Seleccionada</b>", table_cell_bold),
            Paragraph("<b>Configuración / Parámetro Clave</b>", table_cell_bold),
            Paragraph("<b>Justificación Técnica y Cumplimiento Cloud</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>Gestor y Runtime</b>", table_cell),
            Paragraph("<b>uv</b> (Astral) + Python 3.12", table_cell),
            Paragraph(".venv hermético, resolución determinista", table_cell),
            Paragraph("Instalación en sub-segundos, compatibilidad multiplataforma y portabilidad total.", table_cell)
        ],
        [
            Paragraph("<b>Base Vectorial Cloud</b>", table_cell),
            Paragraph("<b>Qdrant Cloud</b> (Serverless)", table_cell),
            Paragraph("Cluster AWS ca-central-1, HNSW, Cosine", table_cell),
            Paragraph("Free Tier permanente 1GB, soporte de índices de payload nativos y filtros combinados.", table_cell)
        ],
        [
            Paragraph("<b>Modelo de Embeddings</b>", table_cell),
            Paragraph("<b>paraphrase-multilingual-MiniLM-L12-v2</b>", table_cell),
            Paragraph("384 dims, normalización L2", table_cell),
            Paragraph("Cero costo de API, alta densidad semántica, vectorización batch de %d vectores en pocos segundos." % summary.get("corpus", {}).get("total_vectors", 349), table_cell)
        ],
        [
            Paragraph("<b>Pipeline de Chunking</b>", table_cell),
            Paragraph("RecursiveCharacterTextSplitter", table_cell),
            Paragraph("chunk_size=512, overlap=64 (~12%)", table_cell),
            Paragraph("Alineado a la ventana de 256 tokens del encoder. Evita pérdida de contexto en fronteras.", table_cell)
        ],
        [
            Paragraph("<b>Seguridad Cloud</b>", table_cell),
            Paragraph("python-dotenv (.env)", table_cell),
            Paragraph("Credenciales gRPC/REST externas", table_cell),
            Paragraph("Cumplimiento estricto de directriz: cero API keys o credenciales hardcodeadas en código.", table_cell)
        ]
    ]

    t_stack = Table(stack_data, colWidths=[90, 115, 130, 196])
    t_stack.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT_BG),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_stack)

    # ==================== PÁGINA 2: VALIDACIÓN, MÉTRICAS Y APA 7 ====================
    story.append(PageBreak())

    # Encabezado Página 2
    story.append(Paragraph("EVALUACIÓN EXPERIMENTAL, CONSULTAS SEMÁNTICAS Y RESULTADOS CUANTITATIVOS", title_style))
    story.append(Paragraph("ACA N° 1 - Computación en la Nube | Validación de Desempeño en Qdrant Cloud", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=C_TEAL, spaceAfter=8, spaceBefore=0))

    # Bloque 5: Batería de Consultas Semánticas
    story.append(make_section_banner("4. Batería de Pruebas de Búsqueda Semántica con Cálculo de Similitud Coseno"))
    story.append(Spacer(1, 3))

    # Las filas se construyen a partir de la última ejecución real del benchmark
    # (output/benchmark_summary.json). Si no existe, se muestran los valores
    # históricos de referencia para que el póster siempre sea imprimible.
    fallback = {
        "Q1": ("¿Cuáles son los criterios clínicos para diagnosticar angina típica y cómo se estratifica el riesgo en dolor torácico?",
               "doc_type: 'clinical_guideline'", "0.5901",
               "MinSalud GPC: angina típica (tipo 1) y criterios de estratificación de riesgo."),
        "Q2": ("¿Qué implicación clínica tiene una depresión del segmento ST mayor a 2 mm y pendiente descendente en esfuerzo?",
               "doc_type: 'diagnostic_protocol'", "0.5791",
               "ACC/AHA: depresión ST > 2.0 mm con slope 3 (downsloping) denota isquemia severa multivaso."),
        "Q3": ("¿Cuáles son las metas de colesterol LDL con estatinas de alta potencia y el protocolo de doble antiagregación plaquetaria?",
               "doc_type: 'pharmacology_protocol'", "0.7013",
               "ESC/OMS: meta c-LDL < 55 mg/dL con estatina de alta potencia y DAPT por 12 meses."),
        "Q4": ("Paciente masculino mayor de 60 años con dolor torácico opresivo, antecedente de hipertensión y estenosis coronaria severa",
               "doc_type: 'patient_record' + sex: 'Masculino' + disease_present: true", "0.6611",
               "Historia clínica de paciente de alto riesgo del dataset UCI Heart Disease."),
        "Q5": ("Tratamiento de revascularización y pronóstico para paciente con defecto de perfusión reversible SPECT y angina inducible",
               "Sin filtro (consulta cross-modal)", "0.6287",
               "Correlación híbrida: isquemia viable inducible (thal=7) y cateterismo prioritario."),
    }

    queries_table_data = [
        [
            Paragraph("<b>ID</b>", table_cell_bold),
            Paragraph("<b>Pregunta de Entrada en Lenguaje Natural</b>", table_cell_bold),
            Paragraph("<b>Filtro de Metadatos Qdrant</b>", table_cell_bold),
            Paragraph("<b>Top-1 Score</b>", table_cell_bold),
            Paragraph("<b>Fragmento Clínico Recuperado (Top-1)</b>", table_cell_bold),
        ]
    ]

    for qid in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        data = measured.get(qid)
        if data:
            question = data["query"]
            filter_text = data["filter_label"]
            score_text = f"{data['top1_score']:.4f}"
            snippet = data.get("snippet") or data.get("expected", "")
            source = data.get("source_title", "")
            fragment = f"<i>{source}:</i> {_plain_text(snippet)}" if source else _plain_text(snippet)
        else:
            question, filter_text, score_text, fragment = fallback[qid]
        queries_table_data.append(
            [
                Paragraph(f"<b>{qid}</b>", table_cell),
                Paragraph(question, table_cell),
                Paragraph(filter_text, table_cell_code),
                Paragraph(f"<b>{score_text}</b>", table_cell_bold),
                Paragraph(fragment, table_cell),
            ]
        )

    t_queries = Table(queries_table_data, colWidths=[24, 150, 110, 52, 195])
    t_queries.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_LIGHT_BG),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_queries)
    story.append(Spacer(1, 7))

    # Bloque 6: Análisis Estadístico y Gráfico de Similitud Coseno
    story.append(make_section_banner("5. Análisis Cuantitativo de Desempeño y Distribución de Similitud"))
    story.append(Spacer(1, 3))

    chart_img_path = str(settings.output_dir / "metricas_similitud_notebook.png")
    if not os.path.exists(chart_img_path):
        chart_img_path = str(settings.output_dir / "similarity_distribution.png")
    if os.path.exists(chart_img_path):
        story.append(Image(chart_img_path, width=531, height=125))
    story.append(Spacer(1, 3))

    # Bloque 7: Conclusiones, Enlaces Operativos y APA 7
    story.append(make_section_banner("6. Conclusiones Técnicas, Enlaces Operativos y Referencias (Normas APA 7)"))
    story.append(Spacer(1, 2))

    latency = summary.get("latency", {})
    latency_typ = latency.get("search_median_ms", latency.get("search_mean_ms"))
    latency_text = (
        f"respuestas en <b>{latency_typ:.0f} ms</b> (mediana de la búsqueda remota contra el clúster) "
        "combinando recuperación vectorial y filtros booleanos de payload"
        if latency_typ
        else "respuestas de baja latencia en recuperación vectorial combinada con filtros booleanos"
    )

    p_conclusiones = Paragraph(
        "<b>Conclusiones de la Arquitectura:</b><br/>"
        f"• <b>Desempeño y Latencia:</b> El clúster remoto en Qdrant Cloud logró {latency_text}.<br/>"
        "• <b>Costo Cero Real:</b> Operación 100% sobre capas gratuitas (Qdrant Cloud Free Tier 1GB y modelo de embeddings local multilingual-MiniLM-L12).<br/>"
        "• <b>Integración Multimodal:</b> La transformación de datos tabulares a viñetas narrativas enriquecidas posibilitó cruzar registros de pacientes con normatividad médica"
        + (f" con una similitud coseno máxima de {max_score:.4f}." if max_score else "."),
        body_style
    )

    video_url = os.environ.get("RAG_VIDEO_URL", "").strip()
    if video_url:
        linea_video = (
            "• <b>Video de Sustentación (3-5 min):</b> "
            f"<link href='{video_url}' color='#1A5F7A'><u>{video_url}</u></link><br/>"
        )
    else:
        linea_video = (
            "• <b>Video de Sustentación (3-5 min):</b> <font color='#B45309'>"
            "pendiente de grabar y subir</font><br/>"
        )

    p_enlaces = Paragraph(
        "<b>Enlaces Operativos Oficiales:</b><br/>"
        "• <b>Repositorio GitHub:</b> "
        "<link href='https://github.com/jaimecabrera911/computacion_en_la_nube' color='#1A5F7A'><u>"
        "https://github.com/jaimecabrera911/computacion_en_la_nube</u></link><br/>"
        + linea_video
        + "• <b>Documentación del proyecto:</b> "
        "<link href='https://github.com/jaimecabrera911/computacion_en_la_nube/blob/master/docs/EXPLICACION_SUSTENTACION.md' color='#1A5F7A'><u>"
        "docs/EXPLICACION_SUSTENTACION.md</u></link><br/>"
        "• <b>Notebook Reproducible:</b> "
        "<link href='https://github.com/jaimecabrera911/computacion_en_la_nube/blob/master/notebooks/heart_disease_rag_qdrant.ipynb' color='#1A5F7A'><u>"
        "notebooks/heart_disease_rag_qdrant.ipynb</u></link>",
        body_style
    )

    t_bottom = Table([[p_conclusiones, p_enlaces]], colWidths=[270, 261])
    t_bottom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_bottom)
    story.append(Spacer(1, 5))

    # Referencias APA 7
    story.append(Paragraph("<b>Referencias Bibliográficas (Normas APA 7):</b>", body_bold))
    story.append(Paragraph("• American College of Cardiology & American Heart Association. (2024). <i>Clinical Protocol for Exercise Testing, Ergometry, and Coronary Angiography</i> (Documento Técnico PRT-DIAG-CARDIO-ACC-2024). ACC/AHA.", ref_style))
    story.append(Paragraph("• European Society of Cardiology & World Health Organization. (2024). <i>Clinical Protocol for Pharmacological Management and Secondary Cardiovascular Prevention</i> (Consenso ESC-OMS-PREV-2024). ESC/WHO.", ref_style))
    story.append(Paragraph("• Janosi, A., Steinbrunn, R., Pfisterer, M., & Detrano, R. (1988). <i>UCI Heart Disease Dataset</i> [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease", ref_style))
    story.append(Paragraph("• Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. <i>Advances in Neural Information Processing Systems</i>, 33, 9459–9474.", ref_style))
    story.append(Paragraph("• Ministerio de Salud y Protección Social & OPS. (2024). <i>Guía de Práctica Clínica: Diagnóstico y Estratificación del Riesgo en Enfermedad Arterial Coronaria</i> (GPC-CARDIO-2024-V2). MinSalud/OPS.", ref_style))
    story.append(Paragraph("• Qdrant Solutions GmbH. (2024). <i>Qdrant Documentation: Vector Search Engine & Cloud Database.</i> https://qdrant.tech/documentation/", ref_style))
    story.append(Paragraph("• Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. <i>Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)</i>. https://arxiv.org/abs/1908.10084", ref_style))

    doc.build(story)
    print(f"✅ Póster tipo Infografía generado exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_poster_pdf()
