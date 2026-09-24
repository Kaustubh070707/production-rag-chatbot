from fastapi import FastAPI, UploadFile, Request, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app import chunking,retriever,bm25_retriever,hybrid,llm
import logging,time,os
from collections import defaultdict, deque

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="RAG Chatbot — Hybrid Retrieval Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MIN_SCORE = 0.85
RAW_DENSE_FLOOR = 0.2
RAW_BM25_FLOOR = 1.0
DOCS_DIR = Path("docs")
# simple in-memory rate limit: 20 requests per 60s per IP on /ask
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
_WINDOW = 60
_hits: dict[str, deque] = defaultdict(deque)
# repeat-query cache: ~5 min TTL, saves LLM cost on popular questions
_CACHE_TTL = 300
_cache: dict[str, tuple[float, dict]] = {}

class AskRequest(BaseModel):
    query: str
    top_k: int = 2


def load_documents() -> list[tuple[str,str]]:
    pairs:list[tuple[str,str]] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunking.chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
            pairs.append((path.name,chunk))
    return pairs

@app.get("/", include_in_schema=False)
def root_page():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest, request: Request):
    # repeat-query cache (before rate limit, saves cost)
    cache_key = f"{req.query.strip().lower()}::{req.top_k}"
    cached = _cache.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1]
    # rate limit by IP
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    dq = _hits[ip]
    while dq and now - dq[0] > _WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"Rate limit: {_RATE_LIMIT} requests per minute")
    dq.append(now)
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
    # top_chunk,top_score = results[0]
    # top_source = source_by_chunk[top_chunk]
    # answer = f"Top match from {top_source}: {top_chunk[:500]}"

    # return {"query":req.query, "answer":answer,"citations":citations}


    cited = [(source_by_chunk[chunk],chunk) for chunk,_ in results[:2]]
    metrics:dict[str,int] = {}
    t0 = time.time()

    try:
        answer = llm.generate_answer(req.query,cited,metrics)
        llm_latency_ms = round((time.time()-t0)*1000,1)
    except Exception as e:
        logging.exception("LLM failed: falling back to extractive %s", e)
        top_chunk, _ = results[0]
        answer = f"Top match from {source_by_chunk[top_chunk]}: {top_chunk[:500]}"
        llm_latency_ms = None

    if "prompt_tokens" in metrics and "completion_tokens" in metrics:
        llm_tokens = metrics["prompt_tokens"] + metrics["completion_tokens"]
    else:
        llm_tokens = "TBD"

    resp = {
        "query": req.query,
        "answer": answer,
        "citations":citations,
        "llm_latency_ms":llm_latency_ms,
        "llm_tokens":llm_tokens,
    }
    # cache successful answers (including refusals — they are cheap and stable)
    if len(_cache) > 500:
        _cache.clear()
    _cache[cache_key] = (time.time(), resp)
    return resp


    # TODO Step 5: hybrid + re-rank, Step 6: citations + refusal

@app.post("/ingest")
async def ingest(file: UploadFile, request: Request):
    if os.getenv("DISABLE_INGEST", "false").lower() == "true":
        raise HTTPException(status_code=503, detail="Ingest disabled on demo deployment — run locally to upload")
    # optional shared-secret auth for live: set RAG_API_KEY on server, clients send X-API-Key
    expected = os.getenv("RAG_API_KEY")
    if expected:
        provided = request.headers.get("x-api-key")
        if provided != expected:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    # TODO: parse PDF/DOCX/HTML/MD -> chunk -> embed -> pgvector
    # Move to background worker (Celery/BullMQ/RQ) to keep upload <300ms
    return {"filename": file.filename, "status": "queued"}
