from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from backend import FinTechRAG

load_dotenv()

app = FastAPI(title="FinTech RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system: Optional[FinTechRAG] = None


class QuestionRequest(BaseModel):
    question: str
    max_sources: int = 3


class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str


@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = FinTechRAG(
        data_path=os.getenv("DATA_PATH"),
        vector_db_path=os.getenv("VECTOR_DB_PATH")
    )


@app.get("/")
async def root():
    return {"status": "API running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):

    global rag_system

    try:

        if rag_system.qa_chain is None:
            rag_system.initialize()

        result = rag_system.ask(request.question)

        sources = result["sources"][:request.max_sources]

        return AnswerResponse(
            answer=result["answer"],
            sources=sources,
            query=result["query"]
        )

    except Exception as e:
        print("API ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}



