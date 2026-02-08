import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_agent_groq import GroqRAGAgent

# Load env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# FastAPI App
# ==============================
app = FastAPI(
    title="Groq RAG API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Schema
# ==============================
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# ==============================
# Global Agent
# ==============================
agent: GroqRAGAgent | None = None

# ==============================
# Startup
# ==============================
@app.on_event("startup")
async def startup():
    global agent
    logger.info("Initializing Groq RAG Agent...")
    agent = GroqRAGAgent()
    logger.info("Groq RAG Agent ready")

# ==============================
# Routes
# ==============================
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Incoming query: {request.query[:50]}...")

    answer = agent.answer(request.query)
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok"}
