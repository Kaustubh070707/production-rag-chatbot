# D1 Production RAG Chatbot Over Your Own Documents

> Track D / Beginner-Intermediate / 3 weeks. RAG that answers strictly from your corpus with citations + eval.

## What it is
FastAPI RAG service: ingest PDF/DOCX/HTML/MD -> chunk + embed -> pgvector + hybrid BM25 -> re-rank -> grounded answer with refusal.

## Why it strengthens profile
Most in-demand applied-AI skill. Eval set + chunking comparison + cost tracking puts you in top tier vs notebook-only candidates.

## Build order (from vault - do in this order)
1. Naive version: fixed chunk, embed, top-5, stuff into prompt. End-to-end first.
2. Write 30 test Qs with known answers (`eval/questions.jsonl`). Baseline hit-rate.
3. Measure hit-rate. Record baseline.
4. Experiment chunk size/overlap. Re-measure. Table below.
5. Add hybrid + re-rank. Re-measure.
6. Add citations + strict refuse prompt. Test unanswerable.
7. Add cache + token/cost tracking.
8. FastAPI + UI + Docker + deploy + demo video.

## Results table
| Stage | Hit-rate | Notes |
|---|---|---|
| baseline fixed chunk | TBD | |
| tuned chunk/overlap | TBD | |
| hybrid + re-rank | TBD target 89% | |

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# docs at http://localhost:8000/docs
```

## Interview gate (must answer before resume)
1. How do you know it's good? Number + method?
2. Chunk 500 - what breaks at 100 and 2000?
3. How did you stop answering uncovered questions?

## Resume bullet template
Built production RAG assistant (FastAPI, pgvector, hybrid BM25 + vector with re-rank) over 2k-doc corpus; raised hit-rate from 61% to 89% on 30-Q set and cut per-query cost by caching.
