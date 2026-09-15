from rank_bm25 import BM25Okapi

STOPWORDS = {"who", "was", "the", "a", "an", "of", "in", "on",
             "and", "or", "to", "for", "with", "what",
             "how", "when", "where", "explain", "tell", "why",
             "is", "are", "were", "do", "does", "did",
             "can", "could", "should"}


def tokenize(text:str)->list[str]:
    temp:str = ""
    text_lst = []
    for ch in text.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            temp += ch
        else:
            if len(temp) > 2 and temp not in STOPWORDS:
               text_lst.append(temp)
            temp = ""

    if len(temp) > 2 and temp not in STOPWORDS:
        text_lst.append(temp)        
    return text_lst

def bm25_retrieve(query:str,chunks:list[str],top_k:int = 3)->list[tuple[str,float]]:
    corpus = [tokenize(c) for c in chunks]
    qtokens = tokenize(query)
    top_k = max(top_k,0)

    if not qtokens:
        return [(c,0.0) for c in chunks[:top_k]]

    bm25 = BM25Okapi(corpus)
    scores: list[float] = [float(s) for s in bm25.get_scores(qtokens)]
    ranked:list[tuple[str,float]] = sorted(zip(chunks,scores), key=lambda t:t[1],reverse=True)
    return ranked[:top_k]
