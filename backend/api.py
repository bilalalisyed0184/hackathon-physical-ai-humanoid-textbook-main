from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_agent_groq import GroqRAGAgent

app = FastAPI(title="Groq RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = GroqRAGAgent()

class QueryRequest(BaseModel):
    query: str = ""  # default empty string to avoid None issues

@app.post("/ask")
async def ask(data: QueryRequest):
    query_text = data.query.strip() if data.query else ""
    if not query_text:
        # fallback greeting if empty
        return {
            "answer": "Hi 👋 How can I help you?\nThis textbook contains chapters 1-5. You can ask questions from them 😊"
        }

    answer = agent.answer(query_text)
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok"}
