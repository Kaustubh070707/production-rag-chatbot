from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pathlib import Path
from app import chunking,retriever,bm25_retriever,hybrid

app = FastAPI(title="Production RAG Chatbot - D1")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MIN_SCORE = 0.85
RAW_DENSE_FLOOR = 0.2
RAW_BM25_FLOOR = 1.0
DOCS_DIR = Path("docs")

class AskRequest(BaseModel):
    query: str
    top_k: int = 5


def load_documents() -> list[tuple[str,str]]:
    pairs:list[tuple[str,str]] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunking.chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
            pairs.append((path.name,chunk))
    return pairs

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    pairs = load_documents()
    if not pairs:
        return {"query": req.query,"answer": "No documents loaded.","citations":[]}

    text = [chunk for _,chunk in pairs]
    # results = retriever.retrieve(req.query,text,top_k=max(req.top_k,0))
    # results = bm25_retriever.bm25_retrieve(req.query,text,top_k=max(req.top_k,0))
    results,raw_dense,raw_bm25 = hybrid.hybrid_retrieve_with_raw(
        req.query,text,top_k=max(req.top_k,0)
    )
    # if not results or results[0][1] == 0.0:
    #     return {"query": req.query, "answer": "Not found in your documents.", "citations": []}

    if raw_dense < RAW_DENSE_FLOOR or raw_bm25 < RAW_BM25_FLOOR:
        return {"query": req.query, "answer": "Not found in your documents.", "citations": []}



    if not results or results[0][1] < MIN_SCORE:
        return {"query":req.query,"answer":"Not found in your documents.", "citations":[]}

    source_by_chunk = {chunk:src for src,chunk in pairs}

    citations = [
        {"source":source_by_chunk[chunk],"score":score,"chunk":chunk[:300]}
        for chunk,score in results
    ]
    top_chunk,top_score = results[0]
    top_source = source_by_chunk[top_chunk]
    answer = f"Top match from {top_source}: {top_chunk[:500]}"

    return {"query":req.query, "answer":answer,"citations":citations}

    # TODO Step 5: hybrid + re-rank, Step 6: citations + refusal

@app.post("/ingest")
async def ingest(file: UploadFile):
    # TODO: parse PDF/DOCX/HTML/MD -> chunk -> embed -> pgvector
    # Move to background worker (Celery/BullMQ/RQ) to keep upload <300ms
    return {"filename": file.filename, "status": "queued"}
