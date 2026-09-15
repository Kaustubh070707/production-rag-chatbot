STOPWORDS = {"who", "was", "the", "a", "an", "of", "in", "on",
             "and", "or", "to", "for", "with", "what"}

def tokenizer(text:str)->dict[str,int]:
    temp:str = ""
    text_dict = {}
    for ch in text.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            temp += ch
        else:
            if len(temp) > 2 and temp not in STOPWORDS:
                text_dict[temp] = text_dict.get(temp,0) + 1
            temp = ""

    if len(temp) > 2 and temp not in STOPWORDS:
        text_dict[temp] = text_dict.get(temp,0) + 1        
    return text_dict

def get_score(query_map:dict[str,int],chunk_map:dict[str,int])->float:

    if not query_map or not chunk_map:
        return 0.0

    common = sum(query_map[w] for w in query_map if w in chunk_map)
    return common/sum(query_map[w] for w in query_map)

def retrieve(query:str,chunks:list[str],top_k:int = 3)->list[tuple[str,float]]:
    scores = []

    query_dict:dict[str,int] = tokenizer(query)
    for x in chunks:
        chunk_dict:dict[str,int] = tokenizer(x)
        score:float = get_score(query_dict,chunk_dict)
        scores.append((x,score))

    scores.sort(key=lambda t:t[1],reverse=True)
    return scores[:top_k]

