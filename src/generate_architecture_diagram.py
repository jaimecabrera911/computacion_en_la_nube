import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram(output_path: str = "output/cloud_rag_architecture.png"):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    def draw_box(x, y, w, h, title, subtitle, color, border_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=2.0",
            facecolor=color,
            edgecolor=border_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 5, title, ha='center', va='top', fontsize=11, fontweight='bold', color='#0F172A', zorder=3)
        ax.text(x + w/2, y + h/2 - 2, subtitle, ha='center', va='center', fontsize=8.5, color='#334155', zorder=3, multialignment='center')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#0284C7", lw=2.2),
            zorder=4
        )
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2.5, label, ha='center', va='bottom', fontsize=8, fontweight='bold', color="#0369A1", zorder=5)

    # 1. Ingestion Sources
    draw_box(3, 55, 26, 36, "1. INGESTA MULTIMODAL", 
             "• UCI Heart Disease (ID: 45)\n  (303 pacientes, 14 atributos)\n• 3 Guías Clínicas (PDF/MD)\n  MinSalud, ACC/AHA, ESC/OMS",
             "#E0F2FE", "#0284C7")

    # 2. Chunking & Enrichment
    draw_box(36, 55, 26, 36, "2. CHUNKING & METADATOS",
             "• Viñetas Clínicas Estructuradas\n  (Estratificación de Riesgo)\n• RecursiveCharacterTextSplitter\n  (chunk_size=512, overlap=64)\n• 349 Fragmentos Totales",
             "#E0E7FF", "#4F46E5")

    # 3. Embedding Model
    draw_box(70, 55, 26, 36, "3. EMBEDDINGS VECTORIALES",
             "• multilingual-MiniLM-L12\n• Dense 384-d latent space\n• Normalización L2 (Dot=Cosine)\n• Inserción Batch (size=64)\n• Costo Cero / Zero Quota Fail",
             "#FEF3C7", "#D97706")

    # 4. Qdrant Cloud Cluster (Center bottom)
    draw_box(30, 8, 40, 36, "4. QDRANT CLOUD (AWS ca-central-1)",
             "• Clúster Serverless Managed\n• Colección: clinical_heart_disease_rag\n• Métrica: Cosine Distance (HNSW)\n• Payload Indexes: doc_type, sex, risk_level, disease_present\n• Latencia de Retrieval: < 150 ms",
             "#DCFCE7", "#16A34A")

    # 5. Semantic Retrieval (Bottom Left)
    draw_box(3, 8, 22, 36, "5. BÚSQUEDA RAG",
             "• Consultas en Lenguaje Natural\n• Top-k Ranking (k=3, 5)\n• Payload Filtering Estricto\n• Inferencia Diagnóstica",
             "#F3E8FF", "#9333EA")

    # 6. Evaluation (Bottom Right)
    draw_box(75, 8, 22, 36, "6. MÉTRICAS CLOUD",
             "• Similitud Coseno Media > 0.65\n• Consistencia de Top-1\n• Alta especificidad diagnóstica\n• Costo operativo: $0.00 USD",
             "#FCE7F3", "#DB2777")

    # Connectors
    draw_arrow(29, 73, 36, 73, "Transform")
    draw_arrow(62, 73, 70, 73, "Vectorize")
    draw_arrow(83, 55, 65, 44, "Batch Upsert")
    draw_arrow(25, 26, 30, 26, "Semantic Query")
    draw_arrow(70, 26, 75, 26, "Metrics & Top-k")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Diagrama de arquitectura generado en: {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
