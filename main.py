# main.py

from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pdf_parser import parse_pdf
from app.policy_chunker import policy_aware_chunk
from app.rag import RAGPipeline
from app.models import PolicyChunk

app = FastAPI(title="Policy-Aware Medical Insurance PDF Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline(
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="llama3.2:3b",
)

current_chunks: List[PolicyChunk] = []

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    try:
        pages = parse_pdf(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to parse PDF. File may be malformed.")

    chunks = policy_aware_chunk(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text or tables found in PDF.")

    pipeline.build_index(chunks)
    if not pipeline.ready:
        raise HTTPException(status_code=500, detail="Failed to build vector index.")

    global current_chunks
    current_chunks = chunks

    table_chunks = sum(1 for c in chunks if c.content_type == "table")
    return {
        "message": "PDF processed successfully.",
        "pages": len(pages),
        "chunks": len(chunks),
        "table_chunks": table_chunks,
    }


@app.get("/ask")
async def ask_question(question: str) -> Dict[str, Any]:
    if not pipeline.ready:
        raise HTTPException(status_code=400, detail="Upload a PDF before asking questions.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    retrieved = pipeline.retrieve(question, top_k=4)
    if not retrieved:
        return {"answer": "I couldn't find the answer in the provided policy context.", "sources": []}

    # If similarity is too low, avoid hallucinations.
    best_score = max(score for _, score in retrieved)
    if best_score < 0.2:
        return {"answer": "I couldn't find the answer in the provided policy context.", "sources": []}

    answer = pipeline.answer(question, retrieved)
    sources = [
        {
            "section": chunk.section,
            "clause": chunk.clause,
            "page": chunk.page,
            "type": chunk.content_type,
        }
        for chunk, _ in retrieved
    ]

    return {"answer": answer, "sources": sources}
