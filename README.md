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
| baseline 800/80 word-match 5-Q | 4/5 (80%) | `python eval/run_eval.py` |
| baseline 800/80 word-match 30-Q | 19/30 (63%) | 18/20 answerable PASS, 1/10 refusal PASS; 2 splits across chunk boundary, 9 false-positive overlaps; `python eval/run_eval.py` |
| chunk 500/50 word-match 30-Q | 18/30 (60%) | 87 chunks, ans 17/20 ref 1/10; smaller loses context; `python eval/run_chunk_compare.py` |
| chunk 800/80 word-match 30-Q | 19/30 (63%) | 54 chunks, ans 18/20 ref 1/10; current pick |
| chunk 1200/100 word-match 30-Q | 19/30 (63%) | 36 chunks, ans 18/20 ref 1/10; larger dilutes, same score fewer chunks |
| BM25 800/80 MIN=0.5 30-Q | 20/30 (67%) | ans 19/20 ref 1/10; `python eval/run_eval_bm25.py` |
| BM25 800/80 MIN=1.0 30-Q | 21/30 (70%) | ans 19/20 ref 2/10; small stopwords set |
| BM25 800/80 expanded-stopwords MIN=1.0 30-Q | 25/30 (83%) | ans 20/20 ref 5/10; current best sparse, `python eval/run_eval_bm25.py` |
| dense 800/80 cos MIN=0.3 30-Q | 26/30 (87%) | ans 16/20 ref 10/10; perfect refusal, loses 4 precise-keyword Qs; `python eval/run_eval_vector.py` |
| dense 800/80 cos MIN=0.4 30-Q | 24/30 (80%) | ans 14/20 ref 10/10 |
| hybrid a=0.5 MIN=0.85 30-Q | 25/30 (83%) | ans 19/20 ref 6/10; per-query min-max caps refusal (refund scores 0.75+); `python eval/run_eval_hybrid.py` |
| hybrid a=0.5 raw-gate OR (dense0.2/bm25 1.0) + MIN=0.85 30-Q | 29/30 (97%) | ans 19/20 ref 10/10; only Brahmagupta split-phrase FAIL; current best |
| llm grounded 5-sample faithfulness | 5/5 grounded, 4/5 clean style | 1/5 preamble leak (Aryabhata run: reasoning preamble + 13 tag-stuffed cites + 1024 cap hit); others 1-2 cites clean; avg ~1024 tokens/Q (prompt ~525 + completion ~499); `python eval/run_faithfulness.py` |
| llm gpt-oss-20b strict prompt 5-sample | 5/5 grounded, 5/5 clean | 0 preamble leaks; per-sentence [notes.md] (10,3,1,1,6 cites); avg ~979 tokens/Q (prompt ~663 + completion ~316); `python eval/run_faithfulness.py` |
| BM25 800/80 MIN=1.5 30-Q | 21/30 (70%) | ans 19/20 ref 2/10; raising further kills nothing more |
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
