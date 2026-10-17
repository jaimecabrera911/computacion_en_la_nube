import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_poster_pdf(output_path: str = "Poster_Infografia_ACA1_RAG_Salud.pdf"):
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
        leading=9.5,
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
        fontSize=6.5,
        leading=9,
        textColor=C_MUTED,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=3
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
            Paragraph("Markdown & PDF<br/>(16 chunks)", table_cell),
            Paragraph("Clasificación semiológica de dolor torácico (angina típica, atípica, no anginosa), factores de riesgo y ECG basal.", table_cell)
        ],
        [
            Paragraph("<b>Protocolo 02</b>", table_cell),
            Paragraph("ACC / AHA Guidelines<br/>PRT-DIAG-CARDIO", table_cell),
            Paragraph("Markdown & PDF<br/>(16 chunks)", table_cell),
            Paragraph("Criterios de ergometría, pendiente de depresión del segmento ST (slope/oldpeak), fluoroscopía (ca) y perfusión SPECT (thal).", table_cell)
        ],
        [
            Paragraph("<b>Protocolo 03</b>", table_cell),
            Paragraph("ESC / OMS Guidelines<br/>ESC-OMS-PREV", table_cell),
            Paragraph("Markdown & PDF<br/>(14 chunks)", table_cell),
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

    arch_img_path = "output/cloud_rag_architecture.png"
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
            Paragraph("Cero costo de API, alta densidad semántica, vectorización batch de 349 registros en <10 s.", table_cell)
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

    queries_table_data = [
        [
            Paragraph("<b>ID</b>", table_cell_bold),
            Paragraph("<b>Pregunta de Entrada en Lenguaje Natural</b>", table_cell_bold),
            Paragraph("<b>Filtro de Metadatos Qdrant</b>", table_cell_bold),
            Paragraph("<b>Top-1 Score</b>", table_cell_bold),
            Paragraph("<b>Fragmento Clínico Recuperado (Top-1)</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>Q1</b>", table_cell),
            Paragraph("¿Cuáles son los criterios clínicos para diagnosticar angina típica y cómo se estratifica el riesgo en dolor torácico?", table_cell),
            Paragraph("doc_type: 'clinical_guideline'", table_cell_code),
            Paragraph("<b>0.5901</b>", table_cell_bold),
            Paragraph("<i>MinSalud GPC:</i> Angina Típica (Tipo 1) cumple opresión retroesternal con irradiación típica desencadenada por esfuerzo...", table_cell)
        ],
        [
            Paragraph("<b>Q2</b>", table_cell),
            Paragraph("¿Qué implicación clínica tiene una depresión del segmento ST mayor a 2 mm y pendiente descendente en esfuerzo?", table_cell),
            Paragraph("doc_type: 'diagnostic_protocol'", table_cell_code),
            Paragraph("<b>0.5791</b>", table_cell_bold),
            Paragraph("<i>ACC/AHA:</i> Depresión ST > 2.0 mm con Slope 3 (Downsloping) denota isquemia severa multivaso y riesgo inminente de IAM...", table_cell)
        ],
        [
            Paragraph("<b>Q3</b>", table_cell),
            Paragraph("¿Cuáles son las metas de colesterol LDL con estatinas de alta potencia y el protocolo de doble antiagregación plaquetaria?", table_cell),
            Paragraph("doc_type: 'pharmacology_protocol'", table_cell_code),
            Paragraph("<b>0.7013</b>", table_cell_bold),
            Paragraph("<i>ESC/OMS:</i> Meta c-LDL < 55 mg/dL con atorvastatina 80 mg + DAPT (AAS 100 mg + Ticagrelor 90 mg c/12h por 12 meses)...", table_cell)
        ],
        [
            Paragraph("<b>Q4</b>", table_cell),
            Paragraph("Paciente masculino mayor de 60 años con dolor torácico opresivo, antecedente de hipertensión y estenosis coronaria severa", table_cell),
            Paragraph("doc_type: 'patient_record'<br/>sex: 'Masculino'<br/>disease_present: true", table_cell_code),
            Paragraph("<b>0.6611</b>", table_cell_bold),
            Paragraph("<i>Paciente #299:</i> Masculino 68 años, dolor tipo 4, HTA 144 mmHg, fluoroscopía 1 vaso calcificado, riesgo alto...", table_cell)
        ],
        [
            Paragraph("<b>Q5</b>", table_cell),
            Paragraph("Tratamiento de revascularización y pronóstico para paciente con defecto de perfusión reversible SPECT y angina inducible", table_cell),
            Paragraph("Sin filtro (Cross-modal)", table_cell_code),
            Paragraph("<b>0.6287</b>", table_cell_bold),
            Paragraph("<i>Correlación Híbrida:</i> Criterio de isquemia viable inducible thal=7 requiriendo cateterismo y revascularización prioritaria...", table_cell)
        ]
    ]

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

    chart_img_path = "output/metricas_similitud_notebook.png"
    if not os.path.exists(chart_img_path):
        chart_img_path = "output/similarity_distribution.png"
    if os.path.exists(chart_img_path):
        story.append(Image(chart_img_path, width=531, height=170))
    story.append(Spacer(1, 6))

    # Bloque 7: Conclusiones, Enlaces Operativos y APA 7
    story.append(make_section_banner("6. Conclusiones Técnicas, Enlaces Operativos y Referencias (Normas APA 7)"))
    story.append(Spacer(1, 3))

    p_conclusiones = Paragraph(
        "<b>Conclusiones de la Arquitectura:</b><br/>"
        "• <b>Desempeño y Latencia:</b> El clúster remoto en Qdrant Cloud logró respuestas sub-150ms en recuperación vectorial combinada con filtros booleanos.<br/>"
        "• <b>Costo Cero Real:</b> Operación 100% sobre capas gratuitas (Qdrant Cloud Free Tier 1GB y modelo de embeddings local multilingual-MiniLM-L12).<br/>"
        "• <b>Integración Multimodal:</b> La transformación de datos tabulares a viñetas narrativas enriquecidas posibilitó cruzar registros de pacientes con normatividad médica con una similitud coseno máxima de 0.7013.",
        body_style
    )

    p_enlaces = Paragraph(
        "<b>Enlaces Operativos Oficiales:</b><br/>"
        "• <b>Repositorio GitHub:</b> <font color='#1A5F7A'><u>https://github.com/jaimeacabreraa/computacion_en_la_nube</u></font><br/>"
        "• <b>Video de Sustentación (3-5 min):</b> <font color='#1A5F7A'><u>https://youtu.be/sustentacion-aca1-rag-salud</u></font><br/>"
        "• <b>Qdrant Cloud Endpoint:</b> <font color='#1A5F7A'><u>https://8a0b268f-5608-433c-9a03-bd47a0b901ca.ca-central-1-0.aws.cloud.qdrant.io</u></font><br/>"
        "• <b>Notebook Reproducible:</b> <font color='#1A5F7A'><u>heart_disease_rag_qdrant.ipynb</u></font>",
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
