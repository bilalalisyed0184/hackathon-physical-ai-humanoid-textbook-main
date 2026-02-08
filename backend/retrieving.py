import logging
import os
import cohere
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

co = cohere.Client(os.getenv("COHERE_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "textbook"


def retrieve_documents(query: str, limit: int = 3):
    try:
        embedding = co.embed(
            texts=[query],
            model="embed-english-v3.0"
        ).embeddings[0]

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
        )

        if not results:
            return []

        return [hit.payload.get("text", "") for hit in results]

    except Exception as e:
        logger.error(f"Qdrant retrieval error: {e}")
        return []
