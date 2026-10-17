import os
from dataclasses import dataclass
from dotenv import load_dotenv
from qdrant_client.http.models import Distance

# Cargar variables de entorno desde archivo .env
load_dotenv()

@dataclass
class Settings:
    # Qdrant Cloud / Local credentials
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "clinical_heart_disease_rag").strip()
    
    # Embedding Model
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()
    vector_dimension: int = 384
    distance_metric: Distance = Distance.COSINE
    
    # Chunking Strategy
    chunk_size: int = 512
    chunk_overlap: int = 64
    
    # Batch indexing
    batch_size: int = 64

settings = Settings()
