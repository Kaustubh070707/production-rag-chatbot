from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Production RAG Chatbot - D1")

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    # TODO Step 1: naive retrieve top-k + stuff into prompt
    # TODO Step 5: hybrid + re-rank, Step 6: citations + refusal
    return {"query": req.query, "answer": "TODO: implement retrieval", "citations": []}

@app.post("/ingest")
async def ingest(file: UploadFile):
    # TODO: parse PDF/DOCX/HTML/MD -> chunk -> embed -> pgvector
    # Move to background worker (Celery/BullMQ/RQ) to keep upload <300ms
    return {"filename": file.filename, "status": "queued"}
