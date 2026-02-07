import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import logging
from rag_agent_groq import RAGAgent


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Import Groq-based RAG agent
from rag_agent_groq import RAGAgent

# ==============================
# FastAPI App
# ==============================
app = FastAPI(
    title="RAG Agent API (Groq)",
    description="API for RAG Agent using Groq LLM with document retrieval",
    version="1.1.0"
)

# CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Schemas
# ==============================
class QueryRequest(BaseModel):
    query: str

class MatchedChunk(BaseModel):
    content: str
    url: str
    position: int
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    matched_chunks: List[MatchedChunk]
    status: str  # success | error | empty
    error: Optional[str] = None
    query_time_ms: Optional[float] = None
    confidence: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# ==============================
# Global Agent
# ==============================
rag_agent: Optional[RAGAgent] = None

# ==============================
# Startup
# ==============================
@app.on_event("startup")
async def startup_event():
    global rag_agent
    logger.info("Initializing Groq RAG Agent...")
    try:
        rag_agent = RAGAgent()
        logger.info("Groq RAG Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Agent: {e}")
        raise RuntimeError("RAG Agent startup failed")

# ==============================
# Routes
# ==============================
@app.post("/ask", response_model=QueryResponse)
async def ask_rag(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")

    logger.info(f"Incoming query: {request.query[:60]}...")

    try:
        result = rag_agent.query(request.query)

        status = "success"
        if not result.get("answer"):
            status = "empty"

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            matched_chunks=[
                MatchedChunk(**chunk) for chunk in result.get("matched_chunks", [])
            ],
            status=status,
            error=result.get("error"),
            query_time_ms=result.get("query_time_ms"),
            confidence=result.get("confidence"),
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Groq RAG Agent API is running"
    )


# ==============================
# Local run
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
