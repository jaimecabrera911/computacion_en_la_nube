import os
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Cargar variables de entorno
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "clinical_heart_disease_rag").strip()
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

def get_client():
    if QDRANT_URL and QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    elif QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(location=":memory:")

def ask_rag(query_text: str, top_k: int = 3, filter_type: str = "all"):
    """
    Ejecuta la búsqueda semántica en Qdrant Cloud y recupera los fragmentos clínicos más relevantes.
    filter_type: 'all', 'guidelines' (solo guías), 'patients' (solo historias de pacientes)
    """
    client = get_client()
    model = SentenceTransformer(MODEL_NAME)
    
    # Configurar filtro según opción
    query_filter = None
    if filter_type == "guidelines":
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="clinical_guideline"))]
        )
    elif filter_type == "diagnostic":
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="diagnostic_protocol"))]
        )
    elif filter_type == "pharmacology":
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="pharmacology_protocol"))]
        )
    elif filter_type == "patients":
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="patient_record"))]
        )
        
    # Vectorizar consulta
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    
    # Búsqueda en el clúster Qdrant Cloud
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter
    )
    
    print("\n" + "="*80)
    print(f"🔎 CONSULTA MÉDICA: {query_text}")
    print(f"🎯 FILTRO APLICADO: {filter_type.upper()}")
    print("="*80)
    
    if not response.points:
        print("⚠️ No se encontraron resultados para los criterios solicitados.")
        return []
        
    for rank, point in enumerate(response.points, 1):
        score = point.score
        doc_type = point.payload.get("doc_type", "N/A")
        title = point.payload.get("title")
        patient_id = point.payload.get("patient_id")
        
        fuente = title if title else f"Historia de Paciente #{patient_id}"
        texto = point.payload.get("text", "")
        
        print(f"\n[{rank}] 🏆 SIMILITUD COSENO: {score:.4f} | TIPO: {doc_type}")
        print(f"    📌 FUENTE: {fuente}")
        if patient_id:
            edad = point.payload.get("age")
            sexo = point.payload.get("sex")
            riesgo = point.payload.get("risk_level")
            print(f"    👤 METADATOS: {edad} años | Sexo: {sexo} | Nivel de Riesgo: {riesgo}")
        print("    📄 FRAGMENTO RECUPERADO:")
        # Indentar texto para legibilidad
        for linea in texto.split("\n"):
            if linea.strip():
                print(f"       {linea.strip()}")
                
    print("\n" + "="*80 + "\n")
    return response.points

def main():
    parser = argparse.ArgumentParser(description="Consola de Búsqueda Semántica en Qdrant Cloud (RAG Salud)")
    parser.add_argument("query", nargs="?", help="Pregunta médica en lenguaje natural entre comillas")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Número de fragmentos a recuperar (Top-k)")
    parser.add_argument(
        "--filter", "-f",
        choices=["all", "guidelines", "diagnostic", "pharmacology", "patients"],
        default="all",
        help="Filtrar por tipo de documento"
    )
    args = parser.parse_args()

    if args.query:
        ask_rag(args.query, top_k=args.top_k, filter_type=args.filter)
    else:
        print("\n🩺 ASISTENTE RAG CLOUD DE SALUD CARDIOVASCULAR (Qdrant Cloud)")
        print("Escriba su consulta médica en lenguaje natural. Escriba 'salir' o 'exit' para terminar.\n")
        while True:
            try:
                user_input = input("❓ Ingrese su consulta médica: ").strip()
                if user_input.lower() in ["salir", "exit", "quit", "q"]:
                    print("¡Hasta luego!")
                    break
                if not user_input:
                    continue
                ask_rag(user_input, top_k=args.top_k, filter_type=args.filter)
            except (KeyboardInterrupt, EOFError):
                print("\n¡Hasta luego!")
                break

if __name__ == "__main__":
    main()
