from app.bm25_retriever import bm25_retrieve
from app.vector_retriever import vector_retrieve

def _normalize(scores:list[float])->list[float]:
    if not scores:
        return []
    lo,hi = min(scores),max(scores)
    if hi == lo:
        return [0.5]*len(scores)
    return [(s-lo)/(hi-lo) for s in scores]



def hybrid_retrieve(query: str, chunks: list[str], top_k: int = 3, alpha: float = 0.5) -> list[tuple[str, float]]:
    if not query.strip() or not chunks:
        return []
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")

    top_k = max(top_k,0)
    full = len(chunks)

    bm25_ranked = bm25_retrieve(query,chunks,top_k=full)
    dense_ranked = vector_retrieve(query,chunks,top_k=full)

    bm25_map = dict(bm25_ranked)
    dense_map = dict(dense_ranked)

    bm25_scores = [bm25_map.get(c,0.0) for c in chunks]
    dense_scores = [dense_map.get(c,0.0) for c in chunks]

    bm25_n = _normalize(bm25_scores)
    dense_n = _normalize(dense_scores)

    hybrid_scores = [alpha * d + (1-alpha)*b for  d,b in zip(dense_n,bm25_n)]
    ranked = sorted(zip(chunks,hybrid_scores), key=lambda t:t[1],reverse=True)
    return ranked[:top_k]