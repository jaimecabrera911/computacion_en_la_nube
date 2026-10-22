import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rag.config import settings

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """
    Inicializa el cliente de Qdrant según las variables de entorno.
    Si QDRANT_URL y QDRANT_API_KEY están presentes en .env, conecta al clúster Cloud.
    En caso contrario, inicializa un almacén local en memoria o persistente para reproducibilidad.
    """
    if settings.qdrant_url and settings.qdrant_api_key:
        logger.info(f"Conectando a Qdrant Cloud en {settings.qdrant_url}...")
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30.0
        )
    elif settings.qdrant_url:
        logger.info(f"Conectando a instancia Qdrant en {settings.qdrant_url}...")
        client = QdrantClient(url=settings.qdrant_url, timeout=30.0)
    else:
        logger.info("Modo Qdrant Local/In-Memory activado (reproducibilidad sin credenciales externas).")
        client = QdrantClient(location=":memory:")
        
    return client

def create_or_recreate_collection(
    client: QdrantClient,
    collection_name: str = settings.collection_name,
    vector_size: int = settings.vector_dimension,
    distance: models.Distance = settings.distance_metric
) -> bool:
    """Crea o recrea la colección vectorial configurada con la métrica de distancia seleccionada."""
    existing_collections = [col.name for col in client.get_collections().collections]
    
    if collection_name in existing_collections:
        logger.info(f"Recreando colección existente: {collection_name}")
        client.delete_collection(collection_name=collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance)
    )
    logger.info(f"Colección '{collection_name}' creada exitosamente (size={vector_size}, distance={distance.name}).")
    
    # Configuración de índices de payload para optimización de filtros en Qdrant Cloud
    payload_indexes = [
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("sex", models.PayloadSchemaType.KEYWORD),
        ("disease_present", models.PayloadSchemaType.BOOL),
        ("risk_level", models.PayloadSchemaType.KEYWORD),
        ("patient_id", models.PayloadSchemaType.INTEGER),
        ("age", models.PayloadSchemaType.INTEGER),
    ]
    for field_name, schema in payload_indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema
            )
            logger.info(f"Índice de payload creado para '{field_name}' ({schema}).")
        except Exception as e:
            logger.warning(f"No se pudo crear índice para {field_name}: {e}")
    return True

def index_corpus_in_batches(
    client: QdrantClient,
    model: SentenceTransformer,
    corpus: List[Dict[str, Any]],
    collection_name: str = settings.collection_name,
    batch_size: int = settings.batch_size
) -> int:
    """
    Vectoriza e indexa el corpus por lotes (batch upsert) para optimizar memoria,
    evitar saturar límites de red y garantizar costo cero / rendimiento óptimo.
    """
    total_items = len(corpus)
    logger.info(f"Iniciando indexación por lotes de {total_items} fragmentos en '{collection_name}'...")
    
    for i in range(0, total_items, batch_size):
        batch = corpus[i : i + batch_size]
        batch_texts = [item["text"] for item in batch]
        
        # Generar embeddings normalizados para que dot product equivalga a cosine similarity
        embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        
        points = []
        for j, (item, vector) in enumerate(zip(batch, embeddings)):
            point_id = i + j + 1
            payload = {
                "text": item["text"],
                **item["metadata"]
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
            )
            
        client.upsert(collection_name=collection_name, points=points)
        
    logger.info(f"Indexación completada: {total_items} vectores insertados en Qdrant.")
    return total_items

def semantic_search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    collection_name: str = settings.collection_name,
    limit: int = 5,
    query_filter: Optional[models.Filter] = None
) -> List[Dict[str, Any]]:
    """
    Ejecuta una búsqueda semántica calculando la similitud coseno entre el vector
    de consulta y los vectores almacenados, aplicando filtros de metadatos opcionales.
    """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter
    )
    
    results = []
    for hit in response.points:
        results.append({
            "id": hit.id,
            "score": float(hit.score),
            "text": hit.payload.get("text", "") if hit.payload else "",
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"} if hit.payload else {}
        })
        
    return results
