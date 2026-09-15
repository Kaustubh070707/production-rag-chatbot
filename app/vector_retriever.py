from app.embeddings import embed_texts
from pathlib import Path
import hashlib
import json
import math

MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DIM = 2048
CACHE_PATH = Path("eval/vectors_cache.json")

def _chunk_hash(chunk:str)->str:
    return hashlib.sha1(chunk.encode("utf-8")).hexdigest()

def _load_cache() -> dict[str,list[float]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError,OSError):
        return {}
    if data.get("model") != MODEL or data.get("dim") != DIM:
        return {}
    vectors = data.get("vectors",{})
    return vectors if isinstance(vectors,dict) else {}

def _save_cache(vectors:dict[str,list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True,exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({"model":MODEL,"dim":DIM,"vectors":vectors},indent=2),
        encoding="utf-8",
    )

def _cosine(q:list[float],c:list[float]) -> float:
    q_norm = math.sqrt(sum(x*x for x in q))
    c_norm = math.sqrt(sum(x*x for x in c))

    if q_norm == 0.0 or c_norm == 0.0:
        return 0.0
    dot = sum(a*b for a,b in zip(q,c))
    return dot/(q_norm*c_norm)



def vector_retrieve(query:str,chunks:list[str],top_k:int = 3)->list[tuple[str,float]]:
    if not query.strip() or not chunks:
        return []

    cache = _load_cache()
    missing = [c for c in chunks if _chunk_hash(c) not in cache]
    if missing:
        new_vectors = embed_texts(missing,"passage")
        for chunk,vec in zip(missing,new_vectors):
            cache[_chunk_hash(chunk)] = vec
        _save_cache(cache)

    qvec = embed_texts([query],"query")[0]

    scores = [
        _cosine(qvec,cache[_chunk_hash(c)]) if _chunk_hash(c) in cache else 0.0
        for c in chunks
    ]

    ranked = sorted(zip(chunks,scores),key= lambda t:t[1],reverse=True)
    return ranked[:max(top_k,0)]