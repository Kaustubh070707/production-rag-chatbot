def chunk_text(text,size=500,overlap=50)->list[str]:
    chunks = []
    if size <= 0:
        raise ValueError("Size must be > 0")

    if overlap < 0 or  overlap >= size:
        raise ValueError(f"overlap must be in [0,{size}]): got {overlap}")

    if not text:
        return []

    step = size - overlap

    n_chunks = (len(text)-overlap + step - 1) // step
    
    for k in range(n_chunks):
        start = k*step
        chunks.append(text[start:start+size])
        
    return chunks