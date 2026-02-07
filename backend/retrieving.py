import os
import json
from typing import List, Dict
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
import logging
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self):
        # Initialize Cohere client
        self.cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

        # Initialize Qdrant client
        qdrant_url = os.getenv(
            "QDRANT_URL",
            "https://e5a23b7f-fae6-4fc3-a806-d274200b0882.us-east4-0.gcp.cloud.qdrant.io"
        )
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "rag_embedding"

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for query text using Cohere
        """
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-multilingual-v3.0",
                input_type="search_query"
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding for query: {e}")
            return []

    def query_qdrant(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Query Qdrant for similar vectors and return results with metadata
        """
        try:
            # Use search_points for Qdrant v1.9+
            search_results = self.qdrant_client.search_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
                with_vector=False
            )

            formatted_results = []
            for result in search_results:
                payload = result.payload or {}
                formatted_results.append({
                    "content": payload.get("content", ""),
                    "url": payload.get("url", ""),
                    "position": payload.get("position", 0),
                    "similarity_score": result.score or 0.0,
                    "chunk_id": result.id,
                    "created_at": payload.get("created_at", "")
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            return []

    def verify_content_accuracy(self, retrieved_chunks: List[Dict]) -> bool:
        """
        Basic validation: check that content and URL exist
        """
        for chunk in retrieved_chunks:
            if not chunk.get("content") or not chunk.get("url"):
                logger.warning(f"Missing content or URL in chunk: {chunk.get('chunk_id')}")
                return False
        return True

    def format_json_response(self, results: List[Dict], query: str, query_time_ms: float) -> str:
        """
        Format retrieval results into JSON
        """
        response = {
            "query": query,
            "results": results,
            "metadata": {
                "query_time_ms": query_time_ms,
                "total_results": len(results),
                "timestamp": time.time(),
                "collection_name": self.collection_name
            }
        }
        return json.dumps(response, indent=2)

    def retrieve(self, query_text: str, top_k: int = 5, threshold: float = 0.0, include_metadata: bool = True) -> str:
        """
        Main retrieval function
        """
        start_time = time.time()
        logger.info(f"Processing retrieval request for query: '{query_text[:50]}...'")

        # Step 1: Embedding
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            error_response = {
                "query": query_text,
                "results": [],
                "error": "Failed to generate query embedding",
                "metadata": {"query_time_ms": (time.time() - start_time) * 1000, "timestamp": time.time()}
            }
            return json.dumps(error_response, indent=2)

        # Step 2: Query Qdrant
        raw_results = self.query_qdrant(query_embedding, top_k, threshold)
        if not raw_results:
            logger.warning("No results returned from Qdrant")

        # Step 3: Optional content verification
        if include_metadata and not self.verify_content_accuracy(raw_results):
            logger.warning("Content verification failed for some results")

        # Step 4: Query time
        query_time_ms = (time.time() - start_time) * 1000

        # Step 5: Format JSON
        return self.format_json_response(raw_results, query_text, query_time_ms)

