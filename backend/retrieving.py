import os
import time
import logging
import cohere
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieving")


class RAGRetriever:
    def __init__(self):
        self.cohere = cohere.Client(os.getenv("COHERE_API_KEY"))

        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.collection = "rag_embedding"

    def embed(self, text: str):
        res = self.cohere.embed(
            texts=[text],
            model="embed-multilingual-v3.0",
            input_type="search_query",
        )
        return res.embeddings[0]

    def retrieve(self, query: str, top_k: int = 5):
        start = time.time()

        vector = self.embed(query)

        try:
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant error: {e}")
            return [], 0.0

        chunks = []
        for r in results:
            payload = r.payload or {}
            chunks.append({
                "content": payload.get("content", ""),
                "url": payload.get("url", ""),
                "position": payload.get("position", 0),
                "similarity_score": r.score,
            })

        return chunks, (time.time() - start) * 1000
