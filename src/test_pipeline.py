import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.dataset_loader import load_heart_disease_dataset, generate_patient_corpus
from src.document_processor import load_unstructured_documents, chunk_unstructured_documents
from src.vector_store import (
    get_qdrant_client,
    create_or_recreate_collection,
    index_corpus_in_batches
)
from src.evaluation import run_retrieval_benchmark, plot_similarity_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("=== PASO 1: Aprovisionamiento y Conectividad Qdrant ===")
    client = get_qdrant_client()
    create_or_recreate_collection(client)
    
    print("\n=== PASO 2: Carga y Estructuración de Datos ===")
    # 2.1 Tabular UCI Heart Disease
    df_raw, meta = load_heart_disease_dataset()
    print(f"Dataset UCI Heart Disease cargado: {len(df_raw)} pacientes.")
    patient_corpus = generate_patient_corpus(df_raw)
    print(f"Viñetas clínicas estructuradas generadas: {len(patient_corpus)}")
    
    # 2.2 Corpus no estructurado
    raw_docs = load_unstructured_documents("data/documents")
    print(f"Documentos complementarios cargados: {len(raw_docs)}")
    chunked_docs = chunk_unstructured_documents(raw_docs)
    print(f"Fragmentos generados tras chunking: {len(chunked_docs)}")
    
    total_corpus = chunked_docs + patient_corpus
    print(f"Corpus total para indexación: {len(total_corpus)} documentos/fragmentos.")
    
    print("\n=== PASO 3 & 4: Generación de Embeddings e Indexación Batch ===")
    print(f"Cargando modelo de embeddings: {settings.embedding_model_name}")
    model = SentenceTransformer(settings.embedding_model_name)
    indexed_count = index_corpus_in_batches(client, model, total_corpus)
    print(f"Indexación exitosa de {indexed_count} vectores.")
    
    print("\n=== PASO 5: Búsqueda Semántica y Evaluación Cuantitativa ===")
    df_eval = run_retrieval_benchmark(client, model, top_k=5)
    print("Muestra de resultados de búsqueda semántica:")
    print(df_eval[["query_id", "rank", "score", "doc_type", "source_title"]].head(10))
    
    plot_similarity_metrics(df_eval, "output/similarity_distribution.png")
    print("\n=== Pipeline ejecutado con éxito total ===")

if __name__ == "__main__":
    main()
