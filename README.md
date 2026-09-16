# Production RAG Chatbot

A retrieval-augmented chatbot that answers strictly from your own documents — with citations, and refusal when the answer isn't there.

Built with FastAPI, hybrid BM25 + dense retrieval, and grounded LLM answers. Live demo link below.

## Demo

- **Live:** <paste Railway URL here>
- **Demo video:** <paste 2-min video link here>
- Try in Swagger: open `/docs` → `POST /ask`

```json
{
  "query": "Who was Aryabhata?",
  "top_k": 2
}
```

```json
{
  "query": "Who was Aryabhata?",
  "answer": "Aryabhata I (476 CE) was the first astronomer... [notes.md]",
  "citations": [
    { "source": "notes.md", "score": 1.0, "chunk": "Aryabhata I (476 CE) was..." }
  ],
  "llm_latency_ms": 1800.5,
  "llm_tokens": 850
}
```

Unanswerable questions return `Not found in your documents.` with empty citations — no hallucinations.

## Architecture

```
[notes.md] -> [chunk 800/80 -> 54 pieces] -> [BM25 + dense vectors]
[query] -> [hybrid retrieve top-2 + raw-score refusal gates] -> [LLM grounded answer + cites]
```

- Chunking 800 chars / 80 overlap (compared 500/1200, kept 800: same accuracy, fewer pieces).
- Hybrid retrieval: BM25 keyword scores plus dense cosine, min-max combined, with raw-score floors that refuse weak matches before ranking.
- Grounded generation: strict context-only prompt, per-sentence `[source]` cites, extractive fallback if the LLM fails.

## Results (measured, same 30 questions throughout)

| Stage | Score |
|---|---|
| Word-match baseline | 19/30 |
| BM25 + stopwords + cutoff | 25/30 |
| Dense cosine | 26/30 |
| Hybrid + raw pre-gate refusal | **29/30** (19/20 answerable, 10/10 refusals) |
| Grounded answers faithfulness sample | 5/5 grounded, per-sentence cites |

LLM answers average ~979 tokens per question. Docker image: 1.9GB naive → 478MB multi-stage slim (~75% smaller), `/health` verified inside the container.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # add NVIDIA_API_KEY + model names
uvicorn app.main:app --reload
# docs at http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t rag:slim .
docker run -p 8000:8000 --env-file .env rag:slim
```

## API

- `GET /health` → `{"status": "ok"}`
- `POST /ask` with `{"query": str, "top_k": int}` → `{"query", "answer", "citations", "llm_latency_ms", "llm_tokens"}`
- `POST /ingest` → file upload (MD supported; PDF/DOCX + background worker roadmap)

## Limitations (honest)

- Single-file markdown corpus; no PDF parsing or vector database yet (in-memory + cached vectors).
- Refusal is keyword+cosine thresholding, not a learned judge — 1 of 30 eval questions still fails on a chunk-boundary split.
- Next: cross-encoder re-rank, repeat-query cache, background ingest worker.

## Repo layout

```
app/            FastAPI service (chunking, BM25, dense, hybrid, LLM)
docs/           corpus (your documents go here)
eval/           30-question eval set (questions.jsonl)
Dockerfile      multi-stage slim (~478MB)
Dockerfile.naive  unoptimized baseline (~1.9GB)
SKILL.md        engineering log: decisions, numbers, failures
DEBUG_LOG.md    full code-level debug history
```
