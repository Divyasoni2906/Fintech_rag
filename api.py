from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FinTech RAG API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⭐ IMPORTANT: DO NOT IMPORT BACKEND HERE
rag_system = None


# ---------- Models ----------

class QuestionRequest(BaseModel):
    question: str
    max_sources: int = 3


class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str


# ---------- Health Routes ----------

@app.get("/")
async def root():
    return {"status": "API running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Lazy Loader ----------

def load_rag_system():
    """
    Lazy loads backend ONLY when needed.
    Prevents Render startup crash.
    """
    global rag_system

    if rag_system is None:
        try:
            print("🔄 Importing backend lazily...")

            # ⭐ Lazy import happens here
            from backend import FinTechRAG

            rag_system = FinTechRAG(
                data_path=os.getenv("DATA_PATH"),
                vector_db_path=os.getenv("VECTOR_DB_PATH")
            )

            print("✅ Backend imported successfully")

        except Exception as e:
            print("❌ Backend import failed:", e)
            raise e

    return rag_system


# ---------- Ask Endpoint ----------

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    try:
        rag = load_rag_system()

        # Initialize RAG only when needed
        if rag.qa_chain is None:
            print("🔄 Initializing RAG system...")
            rag.initialize()

        print("💬 Processing question:", request.question)

        result = rag.ask(request.question)

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"][:request.max_sources],
            query=result["query"]
        )

    except Exception as e:
        print("❌ API ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
