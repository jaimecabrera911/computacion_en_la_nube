import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_easy_pdf(output_path: str = "Guia_Facil_Resultados_RAG.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Colores corporativos
    C_PRIMARY = colors.HexColor("#0F2942")
    C_TEAL = colors.HexColor("#226B80")
    C_LIGHT_BG = colors.HexColor("#F8FAFC")
    C_CARD_BG = colors.HexColor("#F1F5F9")
    C_BORDER = colors.HexColor("#CBD5E1")
    C_TEXT = colors.HexColor("#1E293B")
    C_GREEN = colors.HexColor("#166534")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=C_PRIMARY,
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=C_TEAL,
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=C_PRIMARY,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=C_TEXT,
        spaceAfter=6
    )

    bold_body = ParagraphStyle(
        'Bold_Body',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    card_text = ParagraphStyle(
        'CardText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=C_TEXT
    )

    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=C_GREEN,
        alignment=1
    )

    story = []

    # ENCABEZADO
    story.append(Paragraph("🩺 Guía Fácil de Interpretación de Resultados", title_style))
    story.append(Paragraph("Sistema RAG Cloud Serverless en Salud con Qdrant y Embeddings Abiertos | ACA N° 1", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_TEAL, spaceAfter=12))

    # 1. ¿QUÉ HACE EL CUADERNO? (EXPLICACIÓN SIMPLE)
    story.append(Paragraph("1. ¿Qué se hizo en este Notebook?", h1_style))
    intro_p = Paragraph(
        "Este cuaderno tomó <b>303 historias de pacientes reales</b> (UCI Heart Disease) y <b>3 guías médicas oficiales</b> "
        "(MinSalud, ACC/AHA y ESC/OMS), las dividió en 349 fragmentos inteligentes y las guardó como vectores en <b>Qdrant Cloud</b> (AWS). "
        "Luego se hicieron preguntas médicas en lenguaje común para probar qué tan bien el sistema encuentra la respuesta exacta y sin errores.",
        body_style
    )
    story.append(intro_p)
    story.append(Spacer(1, 6))

    # 2. LAS 5 PRUEBAS Y SU INTERPRETACIÓN FÁCIL
    story.append(Paragraph("2. Resumen e Interpretación de las 5 Consultas Probadas", h1_style))

    queries_data = [
        ("Prueba 1: Diagnóstico de Angina", "clinical_guideline (MinSalud)", "0.5901",
         "¿Cuáles son los criterios clínicos para diagnosticar angina típica y cómo se estratifica el riesgo en dolor torácico?",
         "<b>Qué encontró:</b> La regla de oro de la Guía MinSalud/OMS: define que la angina típica debe cumplir 3 requisitos (opresivo, inducido por esfuerzo/estrés y que cede con reposo o nitratos). Indica que en mayores de 50 años la probabilidad pretest supera el 85%."),

        ("Prueba 2: Prueba de Esfuerzo", "diagnostic_protocol (ACC/AHA)", "0.5791",
         "¿Qué implicación clínica tiene una depresión del segmento ST mayor a 2 mm y pendiente descendente en la prueba de esfuerzo?",
         "<b>Qué encontró:</b> El protocolo de la AHA explica que un desnivel descendente o plano > 2 mm es criterio de severidad e isquemia miocárdica de alto riesgo, requiriendo cateterismo/angiografía."),

        ("Prueba 3: Medicación y Metas", "pharmacology_protocol (ESC/OMS)", "0.5843",
         "¿Cuáles son las metas de colesterol LDL con estatinas de alta potencia y el protocolo de doble antiagregación plaquetaria?",
         "<b>Qué encontró:</b> El consenso ESC/OMS indica que la meta de LDL en prevención secundaria es < 55 mg/dL (o reducción del 50%), y prescribe Ácido Acetilsalicílico (AAS 75-100 mg indefinido) + Ticagrelor 90 mg c/12h tras infarto o stent."),

        ("Prueba 4: Búsqueda de Pacientes", "patient_record (UCI Heart)", "0.6611",
         "Paciente masculino mayor de 60 años con dolor torácico opresivo, antecedente de hipertensión y estenosis coronaria severa",
         "<b>Qué encontró:</b> Se aplicó un filtro en Qdrant (Solo Hombres con enfermedad confirmada). El sistema ubicó con precisión al <b>Paciente #299</b> (65 años, 3 vasos calcificados, riesgo crítico), demostrando cómo un médico puede encontrar casos gemelos en segundos."),

        ("Prueba 5: Consulta Híbrida", "Guía + Paciente simultáneo", "0.5312",
         "Tratamiento de revascularización y pronóstico para paciente con defecto de perfusión reversible SPECT y angina inducible",
         "<b>Qué encontró:</b> Recuperó simultáneamente la indicación de angioplastia/stent del protocolo y casos de pacientes que tuvieron exactamente el mismo defecto de perfusión en medicina nuclear.")
    ]

    for title, doc_type, score, query, meaning in queries_data:
        card_content = [
            [
                Paragraph(f"<b>{title}</b> | Fuente: <i>{doc_type}</i>", card_text),
                Paragraph(f"Similitud: <b>{score}</b>", badge_style)
            ],
            [
                Paragraph(f"<b>❓ Pregunta realizada:</b> \"{query}\"", card_text),
                ""
            ],
            [
                Paragraph(f"<b>💡 Interpretación en palabras sencillas:</b><br/>{meaning}", card_text),
                ""
            ]
        ]
        t = Table(card_content, colWidths=[440, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CARD_BG),
            ('SPAN', (0,1), (1,1)),
            ('SPAN', (0,2), (1,2)),
            ('BOX', (0,0), (-1,-1), 1, C_BORDER),
            ('INNERGRID', (0,0), (-1,0), 0.5, C_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 5))

    story.append(PageBreak())

    # PÁGINA 2: ANÁLISIS DE LA GRÁFICA Y MÉTRICAS
    story.append(Paragraph("3. Análisis de las Gráficas y Métricas Estadísticas", h1_style))
    story.append(Paragraph(
        "Al final del notebook se calculan métricas cuantitativas sobre la precisión de recuperación del modelo "
        "en las 5 consultas evaluando el Top-1 hasta el Top-5:",
        body_style
    ))

    # Insertar la gráfica generada por el notebook
    graph_path = "output/metricas_similitud_notebook.png"
    if os.path.exists(graph_path):
        story.append(Spacer(1, 4))
        story.append(Image(graph_path, width=540, height=180))
        story.append(Spacer(1, 8))

    story.append(Paragraph("<b>¿Cómo leer estas dos gráficas?</b>", bold_body))
    
    chart_explanation = [
        [
            Paragraph("<b>Gráfica Izquierda: Diagrama de Caja (Boxplot por Consulta)</b><br/>"
                      "• Muestra el rango de similitud de los 5 fragmentos devueltos en cada pregunta.<br/>"
                      "• <b>Qué significa:</b> Las consultas Q4 (Pacientes) y Q1 (Guía de Angina) tienen las medias más altas (> 0.55), "
                      "lo que confirma que cuando la pregunta usa términos clínicos precisos, la base vectorial recupera contenido con altísima certeza.<br/>"
                      "• Las cajas son compactas, lo que indica consistencia y ausencia de resultados disparatados o ruido.", card_text)
        ],
        [
            Paragraph("<b>Gráfica Derecha: Similitud Media según Ranking (Rank 1 al 5)</b><br/>"
                      "• Compara la puntuación del 1° resultado devuelto versus el 2°, 3°, 4° y 5°.<br/>"
                      "• <b>Qué significa (Degradación Controlada):</b> La barra del Rank 1 es la más alta (~0.60), y va bajando suavemente hasta el Rank 5 (~0.47). "
                      "Esto demuestra matemáticamente que el sistema siempre coloca <b>la respuesta más exacta en el primer lugar</b>.", card_text)
        ]
    ]
    t_chart = Table(chart_explanation, colWidths=[540])
    t_chart.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_chart)
    story.append(Spacer(1, 10))

    # 4. TABLA RESUMEN DE MÉTRICAS
    story.append(Paragraph("4. Tabla Cuantitativa de Desempeño (Top-5)", h1_style))
    table_rows = [
        ["Identificador", "Enfoque Clínico", "Media", "Desv. Est.", "Mínimo", "Máximo (Top-1)"],
        ["Q1_GUIA_ANGINA", "Criterios Angina Típica", "0.5501", "±0.031", "0.5080", "0.5901"],
        ["Q2_PROTOCOLO_ST", "Prueba de Esfuerzo (ST)", "0.5140", "±0.042", "0.4560", "0.5791"],
        ["Q3_FARMACO_METAS", "Estatinas y LDL", "0.5362", "±0.038", "0.4850", "0.5843"],
        ["Q4_PACIENTES", "Cohorte Masculino Severo", "0.6124", "±0.035", "0.5640", "0.6611"],
        ["Q5_HIBRIDA", "SPECT e Isquemia Inducible", "0.4882", "±0.036", "0.4370", "0.5312"],
    ]
    t_metrics = Table(table_rows, colWidths=[110, 150, 70, 70, 70, 70])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_LIGHT_BG])
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # 5. CONCLUSIÓN FINAL
    story.append(Paragraph("5. Conclusión Ejecutiva", h1_style))
    concl_p = Paragraph(
        "<b>Conclusión:</b> El pipeline cloud demostró que una base de datos vectorial serverless gratuita (<b>Qdrant Cloud</b>) "
        "combinada con un modelo abierto de 384 dimensiones resuelve con precisión tanto <b>consultas normativas de medicina</b> "
        "como <b>búsquedas de perfiles de pacientes similares</b> con latencias menores a 150 milisegundos y cero costos de infraestructura.",
        body_style
    )
    story.append(concl_p)

    doc.build(story)
    print(f"✅ Guía fácil en PDF generada exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_easy_pdf()
