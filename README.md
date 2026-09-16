# Production RAG Chatbot

A retrieval-augmented chatbot that answers strictly from your own documents — with citations, and refusal when the answer isn't there.

Built with FastAPI, hybrid BM25 + dense retrieval, and grounded LLM answers. Live demo link below.

## Demo

- **Live:** https://production-rag-chatbot-production.up.railway.app (try `/health`, then `/docs` → `POST /ask`)
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
| baseline 800/80 word-match 5-Q | 4/5 (80%) | First smoke test on 5 hand-written questions. Proved the loop runs, nothing more. `python eval/run_eval.py` |
| baseline 800/80 word-match 30-Q | 19/30 (63%) | First real baseline: 20 answerable with exact phrases verified in notes plus 10 unanswerable. 18 found, only 1 refusal correct — plain word counting matches common words like India everywhere. Two misses are phrases split across chunk boundaries. |
| chunk 500/50 word-match 30-Q | 18/30 (60%) | 87 small chunks. Loses a point because tiny pieces drop surrounding context. `python eval/run_chunk_compare.py` |
| chunk 800/80 word-match 30-Q | 19/30 (63%) | 54 chunks. Kept as the pick: same best score with almost half the pieces of 500, so faster search for free. |
| chunk 1200/100 word-match 30-Q | 19/30 (63%) | 36 big chunks. Same score, but long chunks bury phrases and slow precise matching. |
| BM25 800/80 MIN=0.5 30-Q | 20/30 (67%) | Swapped counting for BM25, which weighs rare words like Vitasta above common India. Gained one refusal. `python eval/run_eval_bm25.py` |
| BM25 800/80 MIN=1.0 30-Q | 21/30 (70%) | Raised the refusal cutoff to 1.0 with the first small stopwords list. Gained a second refusal, kept all answers. |
| BM25 800/80 MIN=1.5 30-Q | 21/30 (70%) | Pushing the cutoff higher changed nothing further, so 1.0 stayed the pick. |
| BM25 800/80 expanded-stopwords MIN=1.0 30-Q | 25/30 (83%) | Added question words (how, when, explain, is, are…) to stopwords. All 20 answerable found, refusals tripled to 5 of 10. |
| dense 800/80 cos MIN=0.3 30-Q | 26/30 (87%) | Same questions through NVIDIA embeddings (2048-dim) + cosine. Refusals perfect at 10 of 10 because meaning mismatches score near zero, but 4 precise-keyword questions lost to nearby paraphrases. `python eval/run_eval_vector.py` |
| dense 800/80 cos MIN=0.4 30-Q | 24/30 (80%) | Tighter cutoff traded two more answers away for nothing, so 0.3 stayed. |
| hybrid a=0.5 MIN=0.85 30-Q | 25/30 (83%) | Averaged BM25 and dense after per-query rescaling. Kept 19 answers but refusals fell to 6 of 10 — rescaling forces every question's best to 1.0, so weak matches look confident. `python eval/run_eval_hybrid.py` |
| hybrid a=0.5 raw-gate OR (dense 0.2 / bm25 1.0) + MIN=0.85 30-Q | 29/30 (97%) | Added refusal on raw scores before rescaling. Only miss left is Brahmagupta, whose phrase is split across a chunk boundary. Current best. |
| llm grounded 5-sample faithfulness | 5/5 grounded, 4/5 clean style | First generation test: all facts present in citations, but one run leaked reasoning preamble with stuffed tags and hit the token cap. Others clean with 1–2 cites. About 1024 tokens per question. `python eval/run_faithfulness.py` |
| llm gpt-oss-20b strict prompt 5-sample | 5/5 grounded, 5/5 clean | Strict rules plus a worked example killed the preamble: zero leaks, per-sentence [notes.md] cites (10, 3, 1, 1, 6), about 979 tokens per question. |
| docker naive -> slim | 1.9GB -> 478MB (~75% smaller) | Same verified service (`/health ok` inside the container). Multi-stage slim drops the build toolchain the naive image ships. |

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
