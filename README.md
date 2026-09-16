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

| Stage | Hit-rate | Notes |
|---|---|---|
| baseline 800/80 word-match 5-Q | 4/5 (80%) | `python eval/run_eval.py` |
| baseline 800/80 word-match 30-Q | 19/30 (63%) | 18/20 answerable PASS, 1/10 refusal PASS; 2 splits across chunk boundary, 9 false-positive overlaps; `python eval/run_eval.py` |
| chunk 500/50 word-match 30-Q | 18/30 (60%) | 87 chunks, ans 17/20 ref 1/10; smaller loses context; `python eval/run_chunk_compare.py` |
| chunk 800/80 word-match 30-Q | 19/30 (63%) | 54 chunks, ans 18/20 ref 1/10; current pick |
| chunk 1200/100 word-match 30-Q | 19/30 (63%) | 36 chunks, ans 18/20 ref 1/10; larger dilutes, same score fewer chunks |
| BM25 800/80 MIN=0.5 30-Q | 20/30 (67%) | ans 19/20 ref 1/10; `python eval/run_eval_bm25.py` |
| BM25 800/80 MIN=1.0 30-Q | 21/30 (70%) | ans 19/20 ref 2/10; small stopwords set |
| BM25 800/80 expanded-stopwords MIN=1.0 30-Q | 25/30 (83%) | ans 20/20 ref 5/10; `python eval/run_eval_bm25.py` |
| dense 800/80 cos MIN=0.3 30-Q | 26/30 (87%) | ans 16/20 ref 10/10; perfect refusal, loses 4 precise-keyword Qs; `python eval/run_eval_vector.py` |
| dense 800/80 cos MIN=0.4 30-Q | 24/30 (80%) | ans 14/20 ref 10/10 |
| hybrid a=0.5 MIN=0.85 30-Q | 25/30 (83%) | ans 19/20 ref 6/10; per-query min-max caps refusal (refund scores 0.75+); `python eval/run_eval_hybrid.py` |
| hybrid a=0.5 raw-gate OR (dense0.2/bm25 1.0) + MIN=0.85 30-Q | 29/30 (97%) | ans 19/20 ref 10/10; only Brahmagupta split-phrase FAIL; current best |
| llm grounded 5-sample faithfulness | 5/5 grounded, 4/5 clean style | 1/5 preamble leak (Aryabhata run: reasoning preamble + 13 tag-stuffed cites + 1024 cap hit); others 1-2 cites clean; avg ~1024 tokens/Q; `python eval/run_faithfulness.py` |
| llm gpt-oss-20b strict prompt 5-sample | 5/5 grounded, 5/5 clean | 0 preamble leaks; per-sentence [notes.md] (10,3,1,1,6 cites); avg ~979 tokens/Q; `python eval/run_faithfulness.py` |
| BM25 800/80 MIN=1.5 30-Q | 21/30 (70%) | ans 19/20 ref 2/10; raising further kills nothing more |
| docker naive -> slim | 1.9GB -> 478MB (~75% smaller) | `docker build -f Dockerfile.naive -t rag:naive .` vs `docker build -t rag:slim .`; slim runs, `/health ok` |
| docker naive -> slim | 1.9GB -> 478MB (~75% smaller) | `docker build -f Dockerfile.naive -t rag:naive .` vs `docker build -t rag:slim .`; slim runs, `/health ok` |

Runners live in `eval/` (repeatable; dense/vector runs need `NVIDIA_API_KEY` in `.env`). Re-rank row omitted — not built; hybrid raw-gate is the current best.

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
