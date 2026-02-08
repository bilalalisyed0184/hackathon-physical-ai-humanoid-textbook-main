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
    query: str


@app.post("/ask")
async def ask(data: QueryRequest):
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    answer = agent.answer(data.query)
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok"}
