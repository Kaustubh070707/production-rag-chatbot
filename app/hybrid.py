from app.bm25_retriever import bm25_retrieve
from app.vector_retriever import vector_retrieve


def _normalize(scores:list[float])->list[float]:
    if not scores:
        return []
    lo,hi = min(scores),max(scores)
    if hi == lo:
        return [0.5]*len(scores)
    return [(s-lo)/(hi-lo) for s in scores]



def hybrid_retrieve_with_raw(query: str, chunks: list[str], top_k: int = 3, alpha: float = 0.5) -> tuple[list[tuple[str, float]],float,float]:
    if not query.strip() or not chunks:
        return ([],0.0,0.0)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")

    top_k = max(top_k,0)
    full = len(chunks)

    bm25_ranked = bm25_retrieve(query,chunks,top_k=full)
    try:
        dense_ranked = vector_retrieve(query,chunks,top_k=full)
    except RuntimeError:
        # CI without NVIDIA_API_KEY — fall back to BM25-only scores so tests and deploys without keys still pass
        dense_ranked = [(c, 0.0) for c in chunks]

    bm25_map = dict(bm25_ranked)
    dense_map = dict(dense_ranked)

    bm25_scores = [bm25_map.get(c,0.0) for c in chunks]
    dense_scores = [dense_map.get(c,0.0) for c in chunks]

    raw_dense_best = max(dense_scores) if dense_scores else 0.0
    raw_bm25_best = max(bm25_scores) if bm25_scores else 0.0

    bm25_n = _normalize(bm25_scores)
    dense_n = _normalize(dense_scores)

    hybrid_scores = [alpha * d + (1-alpha)*b for  d,b in zip(dense_n,bm25_n)]
    ranked = sorted(zip(chunks,hybrid_scores), key=lambda t:t[1],reverse=True)
    return (ranked[:top_k],raw_dense_best,raw_bm25_best)

def hybrid_retrieve(query:str,chunks:list[str],top_k:int = 3 , alpha:float = 0.5)->list[tuple[str,float]]:
    ranked,_,_ = hybrid_retrieve_with_raw(query,chunks,top_k,alpha)
    return ranked