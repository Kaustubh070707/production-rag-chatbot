# Full Eval Log — 30 tuning questions

This is the lab notebook. The README keeps 6 rows that tell the story; this file keeps all 16.

| Stage | Hit-rate | Notes |
|---|---|---|
| baseline 800/80 word-match 5-Q | 4/5 (80%) | First smoke test. `python eval/run_eval.py` |
| baseline 800/80 word-match 30-Q | 19/30 (63%) | 18/20 answerable, 1/10 refusal; word counting matches common words. `python eval/run_eval.py` |
| chunk 500/50 word-match 30-Q | 18/30 (60%) | 87 small chunks, loses context. `python eval/run_chunk_compare.py` |
| chunk 800/80 word-match 30-Q | 19/30 (63%) | 54 chunks, current pick. |
| chunk 1200/100 word-match 30-Q | 19/30 (63%) | 36 big chunks, long bury phrases. |
| BM25 800/80 MIN=0.5 30-Q | 20/30 (67%) | Rare Vitasta > common India. `python eval/run_eval_bm25.py` |
| BM25 800/80 MIN=1.0 30-Q | 21/30 (70%) | Small stopwords set. |
| BM25 800/80 MIN=1.5 30-Q | 21/30 (70%) | No further gain. |
| BM25 800/80 expanded-stopwords MIN=1.0 30-Q | 25/30 (83%) | All 20 answerable, 5/10 refusals. |
| dense 800/80 cos MIN=0.3 30-Q | 26/30 (87%) | Refusals 10/10, loses 4 precise keywords. `python eval/run_eval_vector.py` |
| dense 800/80 cos MIN=0.4 30-Q | 24/30 (80%) | Tighter trades answers. |
| hybrid a=0.5 MIN=0.85 30-Q | 25/30 (83%) | Per-query rescaling caps refusal. `python eval/run_eval_hybrid.py` |
| hybrid a=0.5 raw-gate OR (dense 0.2 / bm25 1.0) + MIN=0.85 | 29/30 (97%) tuning — 19/20 ans, 10/10 ref; only Brahmagupta split | Current best on tuning set. |
| LLM grounded 5-sample faithfulness | 5/5 grounded, 4/5 clean | One preamble leak before strict prompt. `python eval/run_faithfulness.py` |
| LLM gpt-oss-20b strict prompt 5-sample | 5/5 grounded, 5/5 clean | Per-sentence [notes.md] cites, ~979 tokens/Q. |
| docker naive -> slim | 1.9GB -> 478MB (~75% smaller) | `docker build -f Dockerfile.naive -t rag:naive .` vs `docker build -t rag:slim .` |
