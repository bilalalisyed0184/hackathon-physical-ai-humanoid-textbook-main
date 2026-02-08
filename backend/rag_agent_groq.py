import os
import logging
import requests
from retrieving import RAGRetriever
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag_agent_groq")


CHAPTERS_TEXT = """Hi 👋 How can I help you?

This textbook contains the following chapters:

1. ROS 2 Basics  
2. Python Agents & rclpy  
3. URDF Humanoid Modeling  
4. Simulation Techniques  
5. AI Control Systems  

You can ask questions from these chapters and I’ll help you 😊
"""


class GroqRAGAgent:
    def __init__(self):
        self.retriever = RAGRetriever()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama3-8b-8192"  # ✅ ACTIVE MODEL

    def _call_groq(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful textbook assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )

        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def answer(self, query: str) -> dict:
        query = query.strip()

        # 🔹 Greeting / empty
        if len(query) < 3:
            return self._fallback()

        chunks, time_ms = self.retriever.retrieve(query)

        # 🔹 NOTHING FOUND → chapters guide
        if not chunks:
            return self._fallback()

        context = "\n\n".join(c["content"][:800] for c in chunks)

        prompt = f"""
Answer ONLY from the textbook content below.

Textbook Content:
{context}

Question:
{query}
"""

        try:
            answer = self._call_groq(prompt)
        except Exception as e:
            logger.error(e)
            return self._fallback(error=str(e))

        return {
            "answer": answer,
            "sources": [c["url"] for c in chunks if c["url"]],
            "matched_chunks": chunks,
            "query_time_ms": time_ms,
            "status": "success",
            "confidence": "high",
        }

    def _fallback(self, error: str | None = None):
        return {
            "answer": CHAPTERS_TEXT,
            "sources": [],
            "matched_chunks": [],
            "status": "empty",
            "confidence": "n/a",
            "error": error,
        }
