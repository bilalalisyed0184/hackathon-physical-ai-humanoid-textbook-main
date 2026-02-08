import os
import time
import logging
from dotenv import load_dotenv
from groq import Groq

from retrieving import RAGRetriever

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqRAGAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.retriever = RAGRetriever()

        # session memory (simple)
        self.greeted = False
        self.active_chapter = None

        logger.info("RAG Agent initialized with Groq")

    # =========================
    # Public Entry
    # =========================
    def answer(self, query: str) -> str:
        q = query.lower().strip()

        # 1️⃣ greeting (ONLY ONCE)
        greetings = ["hi", "hello", "hey", "assalam", "salam"]
        if q in greetings and not self.greeted:
            self.greeted = True
            return self._intro_message()

        # 2️⃣ chapter selection
        if any(x in q for x in ["chapter 1", "chap 1", "ros 2", "ros2"]):
            self.active_chapter = "ROS 2 Basics"
            return self._chapter_1_intro()

        # 3️⃣ generic help AFTER chapter selected
        if "help" in q and self.active_chapter:
            return self._help_prompt()

        # 4️⃣ actual RAG answer
        return self._rag_answer(query)

    # =========================
    # Static Responses
    # =========================
    def _intro_message(self) -> str:
        return (
            "Hi 👋 How can I help you?\n\n"
            "This textbook contains the following chapters:\n"
            "1. ROS 2 Basics\n"
            "2. Python Agents & rclpy\n"
            "3. URDF Humanoid Modeling\n"
            "4. Simulation Techniques\n"
            "5. AI Control Systems\n\n"
            "You can ask questions from these chapters 😊"
        )

    def _chapter_1_intro(self) -> str:
        return (
            "Great 👍 You selected **Chapter 1: ROS 2 Basics**.\n\n"
            "This chapter covers:\n"
            "• What is ROS 2\n"
            "• Nodes\n"
            "• Topics\n"
            "• Services\n"
            "• ROS 2 architecture\n\n"
            "What would you like to learn first?"
        )

    def _help_prompt(self) -> str:
        return (
            f"Sure 🙂 Ask me anything from **{self.active_chapter}**.\n\n"
            "For example:\n"
            "• What is a ROS 2 node?\n"
            "• How do ROS 2 topics work?\n"
            "• Difference between ROS 1 and ROS 2?"
        )

    # =========================
    # RAG + Groq
    # =========================
    def _rag_answer(self, query: str) -> str:
        start = time.time()

        retrieval_json = self.retriever.retrieve(query)
        data = eval(retrieval_json)  # safe here (internal)

        chunks = data.get("results", [])

        if not chunks:
            return (
                "Answer not found in the provided documents.\n\n"
                "You can ask questions related to **ROS 2 Basics** chapters 🙂"
            )

        context = "\n\n".join(
            f"- {c['content']}" for c in chunks[:4]
        )

        prompt = f"""
You are a helpful textbook assistant.
Answer ONLY from the given context.

Context:
{context}

Question:
{query}
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq error: {e}")
            return "Sorry, I encountered an error while generating the response."
