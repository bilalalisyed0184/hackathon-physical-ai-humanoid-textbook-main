import os
import logging
from groq import Groq

from retrieving import retrieve_documents
from chapters import default_greeting

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GroqRAGAgent:
    def answer(self, query: str) -> str:
        docs = retrieve_documents(query)

        # 🟢 If nothing found → behave like tutor
        if not docs:
            return default_greeting()

        context = "\n\n".join(docs)

        prompt = f"""
You are a textbook assistant.
Answer ONLY from the context.

Context:
{context}

Question:
{query}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful AI tutor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "Sorry, I couldn’t process your question right now."
