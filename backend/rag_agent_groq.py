# rag_agent_groq.py

import os
import json
import time
import logging
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq
from retrieving import RAGRetriever  # aapka RAG retrieval logic

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

client = Groq(api_key=GROQ_API_KEY)

class RAGAgent:
    def __init__(self, model: str = "llama3-70b-8192"):
        self.model = model
        logger.info("RAG Agent initialized with Groq")

    def query(self, query_text: str) -> Dict:
        start_time = time.time()
        try:
            retriever = RAGRetriever()
            retrieval = retriever.retrieve(query_text=query_text)
            chunks = retrieval.get("retrieved_chunks", [])

            context = "\n\n".join(f"Source: {c['url']}\nContent: {c['content']}" for c in chunks)

            prompt = f"""
You are a helpful assistant.
Answer the question strictly using the context below.
If the answer is not found, say: "Answer not found in the provided documents."

Context:
{context}

Question:
{query_text}
"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": [c["url"] for c in chunks],
                "matched_chunks": chunks,
                "query_time_ms": (time.time() - start_time) * 1000,
                "confidence": self._calculate_confidence(chunks),
            }

        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {
                "answer": "Sorry, an error occurred while generating the response.",
                "sources": [],
                "matched_chunks": [],
                "error": str(e),
                "query_time_ms": (time.time() - start_time) * 1000,
            }

    def _calculate_confidence(self, matched_chunks: List[Dict]) -> str:
        if not matched_chunks:
            return "low"

        avg_score = sum(c.get("similarity_score", 0.0) for c in matched_chunks) / len(matched_chunks)

        if avg_score >= 0.7:
            return "high"
        elif avg_score >= 0.4:
            return "medium"
        return "low"
