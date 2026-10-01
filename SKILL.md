---
project: production-rag-chatbot
track: ai-ml
level: beginner-intermediate
started: 2026-09-09
shipped: 2026-09-25
repo: https://github.com/Kaustubh070707/production-rag-chatbot
live: https://production-rag-chatbot-production.up.railway.app
video: https://youtu.be/i_eClL5WwuQ
---

# 1. What this project is
Non-technical: I have uploaded my Ancient Science notes and ask it questions related to it, it shows answer and file name
Engineer: FastAPI service, split md file into 800-char pieces with 80 overlap, hybrid BM25 plus dense retrieval with raw-score gates, and grounded LLM answers that cite the source and refuse when the answer is not in the docs

# 2. Problem it solves
I waste a lot of time(~20 minutes) finding details e.g. Aryabhata vs Brahmagupta across my notes, this project solves this problem and returns the result in < 1s and refuses if nothing exists

# 3. Architecture
[notes.md - 591 lines ancient science] -> [chunk 800/80 -> 54 pieces] -> [BM25 scores + dense vectors cached in eval/vectors_cache.json]
[query like Who was Aryabhata?] -> [hybrid top-2 with raw gates (dense 0.2 or bm25 1.0) + normalized cutoff 0.85] -> [LLM grounded answer with [notes.md] cites or Not found]

Components:
- I split the corpus into 800-character pieces with 80 overlap, which gives 9 chunks on the current 3-file CC corpus (54 on the previous larger one, where I compared 500/87 vs 1200/36 and kept 800 for speed with no accuracy loss).
- I keep the chunks in a plain Python list and cached dense vectors on disk, scored with hybrid BM25 plus dense cosine. BM25 gives rare words high weight, dense vectors catch meaning like API key versus Key Words. Hybrid keeps the best of both, and the raw gates keep weak matches from looking confident after rescaling.
- I generate answers with a grounded LLM prompt that forces per-sentence [notes.md] cites and an exact refusal string when the context lacks the answer. If the raw gates already refused, I never call the LLM at all, so refused questions cost zero tokens.

# 4. Key decisions and trade-offs
| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Chunking | 500/50 giving 87 chunks vs 800/80 giving 54 vs 1200/100 giving 36 | 800 with 80 overlap | I kept 800 because on my 30-question tuning set it matched the best scores with almost half the chunks of 500, so search is faster with no accuracy loss. | I gave up finer granularity. A phrase like Siddhanta Shiromani split across an 800-char boundary still fails, which 500 sometimes catches. |
| Retrieval | Plain word-overlap vs BM25 alone vs dense alone vs hybrid BM25+dense | Hybrid BM25 plus dense (alpha 0.5, per-query min-max plus raw gates) | Word-match was 19/30, BM25 alone 25/30, dense alone 26/30 but lost 4 precise-keyword questions. Hybrid with raw gates is 29/30 with 10/10 refusals — it keeps BM25 recall and dense meaning. | I gave up simplicity. Hybrid needs two indexes, embedding API calls, caching, and tuning of alpha and two cutoffs. |
| Refusal gates | BM25 MIN 1.0 only vs raw gates (dense 0.2 or bm25 1.0) plus normalized 0.85 | Raw gates before normalizing, then normalized cutoff | Per-query min-max deletes the absolute strength (refund 0.75 looked like Aryabhata 1.0). The raw floors restore it, lifting refusal 6/10 to 10/10 with no answerable loss. | I gave up a single-threshold design. Two gates need tuning on the same 30 questions and the thresholds only transfer to similar corpora. |
| LLM model | nemotron-3-super-120b vs nemotron-3-ultra-550b vs gpt-oss-20b | gpt-oss-20b | On Who was Aryabhata it answered cleanly in about 850 tokens with no preamble and no cap hit, while ultra-550b leaked reasoning and hit 1024 truncated mid-word. With a strict prompt, gpt-oss-20b gives per-sentence cites reliably. | I gave up the larger models' richer phrasing. Ultra's extra quality never survived the strict prompt without leaking planning text. |
| Docker image | naive single-stage python:3.12 vs multi-stage slim | multi-stage slim | Same service ships at 478MB instead of 1.9GB, so pulls and cold starts are faster with identical answers. | Single-file simplicity and a full toolchain inside the image for debugging. |
| OS CVE response | wait for base rebuild vs patch packages in Dockerfile | Patch in Dockerfile (`apt-get install --only-upgrade` the 4 flagged packages) + `docker build --pull` | Waiting left the gate red for days on the same digest: the fixes existed as Debian packages, so the final stage installs them itself instead of waiting for Docker Hub to rebake. `--pull` forces a fresh base-manifest check every run. | Purity of untouched base images and a few extra build seconds. |

# 5. Skills demonstrated
- [x] Embeddings + dense cosine retrieval evidence: `app/embeddings.py` (NVIDIA 2048-dim, query/passage types) + `app/vector_retriever.py` (hash cache in `eval/vectors_cache.json`, ignored) - dense alone 26/30 with 10/10 refusal
- [x] Chunking strategy eval evidence: `README` results table 500/87 vs 800/54 vs 1200/36 chunks with 30-Q scores
- [x] Hybrid retrieval + raw pre-gate evidence: `app/hybrid.py` (`hybrid_retrieve_with_raw`, per-query min-max + raw dense 0.2 / raw BM25 1.0 floors) - 29/30 on tuning set
- [x] Prompt grounding + refusal evidence: `app/llm.py` (strict rules + example, exact `Not found...` refusal, no-preamble rule) wired in `POST /ask` with extractive fallback; refusal pre-gate means no LLM spend on refused queries
- [x] Evaluation methodology evidence: `eval/questions.jsonl` 30-Q tuning set (20 answerable + 10 unanswerable) + held-out `eval/questions_heldout.jsonl` (12 questions never tuned on) + repeatable local runners
- [x] Cost/latency control evidence: `llm_tokens` + `llm_latency_ms` in `/ask` response; 5-sample avg ~979 tokens/Q; raw gates save cost on refusals

# 6. Numbers I measured
| Metric | Before | After | How I measured it |
|---|---|---|---|
| 30-question tuning-set hit-rate with word-match, chunk 800/80, top-2 | Nothing built yet | 19 correct out of 30 (63%) - 18 of 20 answerable found, only 1 of 10 refusals correct | Same 30 questions throughout (tuning set), top-2 citations check or correct Not-found. Runner `python eval/run_eval.py`. |
| 30-question tuning-set hit-rate with BM25, chunk 800/80, expanded stopwords, cutoff 1.0 | 19/30 word-match above | 25 correct out of 30 (83%) - 20 of 20 answerable, 5 of 10 refusals | Same 30, swapped to BM25Okapi with expanded stopwords. The question What is iPhone price in India now refuses instead of matching Delhi via India. |
| Chunk count vs accuracy | 500/50 gave 87 chunks and 18/30, 1200/100 gave 36 chunks and 19/30 | 800/80 gave 54 chunks and 19/30 word-match, chosen as best balance | Same 30, chunked three ways with `python eval/run_chunk_compare.py`. |
| 30-question tuning-set hit-rate with hybrid raw pre-gate, chunk 800/80, alpha 0.5, floors dense 0.25 / bm25 0.3 | 26/30 normalized-only on new CC corpus | 27 correct out of 30 (90%) - 17 of 20 answerable found, 10 of 10 refusals correct | Same 30 rewritten for the 3-file CC corpus (9 chunks score lower absolutely, so floors retuned by grid in `eval/grid_floors.py`). Tuning set — see held-out row below for transfer. |
| Held-out 12-question hit-rate (never tuned on) | — | 10 correct out of 12 (83%) - 5 of 7 answerable found, 5 of 5 refusals correct | New phrasings on new corpus, thresholds frozen. Runner `python eval/run_eval_heldout.py`. Same-corpus held-out is a weak transfer signal, not a guarantee. |
| LLM grounded answers cost and faithfulness on tuning-set sample | No generation, extractive top-match only | 5 out of 5 sampled answers grounded with per-sentence file cites, 0 preamble leaks after strict prompt; average about 806 tokens per question on new corpus | Same 5 answerable questions through `POST /ask` with gpt-oss-20b, checked every sentence ends with a file cite and each fact appears in cited chunks, summed usage tokens. |
| Docker image size | 1.9GB naive single-stage | 478MB multi-stage slim (about 75% smaller) | `docker build -f Dockerfile.naive -t rag:naive .` vs `docker build -t rag:slim .` then `docker images`; both verified with `curl /health` inside container |

# 7. Things that broke and how I fixed them
1. Symptom: I asked the question What is iPhone price in India, which is not in my ancient-science notes, but the system answered with the Delhi iron pillar chunk because the common word India matched. It should have refused with Not found in your documents.
   Cause: My first plain word overlap counted the common word India equal to rare content words, and my stopwords list missed words like how and what variants, so one common match was enough to pass with score 0.33.
   Fix: I switched ranking to BM25 where rare Vitasta weights around 2.5 while common India weights around 0.5, expanded stopwords to include how, when, where, explain, tell, why, is, are, were, do, does, did, can, could, should, and added a cutoff MIN_SCORE of 1.0 in app/main.py so anything below refuses.
   Lesson: I now always test unanswerable questions including tricky overlaps like Delhi, India, and Key, not just obvious refund policy, because happy-path testing hides false positives.
2. Symptom: I asked the question What did Bhaskaracharya compose, expecting Siddhanta Shiromani in citations, but top-2 citations missed the full phrase even though the words were in the file, scoring only 0.33.
   Cause: My 800-character chunk boundary split the phrase so Siddhanta landed at the end of one chunk and Shiromani at the start of the next, so no single chunk contained the full must_contain string I was checking for.
   Fix: I kept 800/80 after comparing sizes but noted the split in README, and verified the same question passes with BM25 re-ranking bringing a better chunk to top-2. Long-term fix is smaller chunks or overlapping shingles for multi-word titles.
   Lesson: I learned that must_contain phrases must fit inside one chunk, otherwise even a correct retriever fails my check. I now verify phrase length against chunk size when writing eval questions.
3. Symptom: I asked the question How to reset API key, which is not in notes, but system answered with Key Words heading chunk scoring 0.25 in word-match and 3.49 in BM25, so it never refused at any cutoff I tried.
   Cause: The word key appears in the heading 16.8 Key Words in notes.md, so any query containing key matches a real heading even though the meaning is completely different from API key.
   Fix: Fixed after this baseline by dense embeddings plus the raw pre-gate. The question How to reset API key now scores dense 0.035-ish and BM25 low on meaning, so the raw floors (dense below 0.2 or BM25 below 1.0) refuse it before ranking, and the 30-question run shows 10 out of 10 refusals correct. The Key Words heading still ranks high on keywords alone, which is documented proof that raw-gate on meaning (dense) was the necessary fix.
   Lesson: Keyword matching cannot tell word sense apart. Dense vectors with a raw cosine floor were the fix.
4. Symptom: I probed my hybrid retriever with two questions. The good question Who was Aryabhata came back with top scores 1.0 and 0.881, which looks right. But the bad question refund policy, which has no answer anywhere in my notes, came back with 0.75 and 0.714. That is far too confident for something that should refuse near zero, and it is why my hybrid refusal is stuck at 6 out of 10 while dense alone gets 10 out of 10.
   Cause: My _normalize function rescales every question by its own best and worst score. For the good question the raw BM25 best was around 2.5, for the bad question only around 0.4, and that 6x gap was exactly the signal telling good apart from bad. Dividing each question by its own max forced both tops to 1.0, so the absolute strength was deleted and only the order within the question survived. Dense raw had the same story with 0.50 versus 0.035 collapsing to two 1.0s. Averaging two pinned tops gives a bad question 0.75, leaving almost no gap for a cutoff to split.
   Fix: I kept per-query normalize for fair alpha weighting in ranking, but added raw gates before normalizing: refuse when raw dense best is below 0.2 or raw BM25 best below 1.0, then normalize survivors for ranking plus a normalized cutoff 0.85. This restored 10/10 refusals with no answerable loss, taking hybrid from 25/30 to 29/30 on the tuning set.
   Lesson: I learned the difference between ranking and calibration. Normalization preserves which chunk is best but erases whether the best is good enough to answer. From now on I tune ranking and refusal on separate signals.

5. Symptom: Trivy blocked the pipeline on 7 Debian HIGHs for days while the base digest never moved — same `sha256:f77ac9e`, same 7 CVEs, every rebuild identical.
   Cause: The fixes existed as Debian packages (`deb13u3`) but Docker Hub had not rebaked `python:3.12-slim`. Waiting on the base meant red forever through no fault of my code, and every adviser suggestion (new tag, `.trivyignore`, exit-code 0) either changed nothing or hid the signal.
   Fix: The final image stage now upgrades the 4 affected packages itself (`apt-get install --only-upgrade libssl3t64 openssl openssl-provider-legacy libpcre2-8-0`, lists removed after), plus `docker build --pull` so base refreshes are caught the day they land.
   Lesson: Distinguish whose bug it is, then fix at that layer. Upstream-stale means patch it yourself and record why; gate-weakening is never the answer to a red gate.

Full code-level history of all 16 early bugs from chunking to BM25 (one-line bodies, loop-variable mutation, dead branches, float64 casts, wiring swaps) is kept in `DEBUG_LOG.md` at the repo root, deliberately outside `docs/` so it never pollutes retrieval.

# 8. What I would do differently at 100x scale
- I would move chunks from in-memory list to `pgvector` with metadata filtering by source and date, plus a background worker for parsing so uploads stay fast, because reloading and re-chunking on every request will not survive restarts or large corpora.
- I would add auth and rate limiting on `POST /ask` and `POST /ingest`, because the live Railway URL currently lets anyone upload files or spend my NVIDIA key. At scale that is a cost and abuse vector.
- I would add a repeat-query cache and cross-encoder re-ranking. Cache cuts cost to zero on popular questions, re-rank fixes the last split-phrase miss (Brahmagupta) that keeps me at 29/30 instead of 30/30.

# 9. Interview answers I have rehearsed
Q: How do you know your RAG is good? Give me a number and how you measured it.
A: On my tuning set (30 questions over a 3-file CC corpus, 9 chunks) I get 27 out of 30 with hybrid raw-gate 800/80 alpha 0.5 — 17 of 20 answerable found and 10 of 10 refusals correct. On a held-out 12 written after freezing thresholds, I get 10 out of 12. Same-corpus held-out is weak transfer, not a guarantee. Previous larger corpus scored 29/30; the smaller CC corpus loses rare-term recall, honestly recorded.
Q: Chunk size 800 - what breaks at 500 and 1200?
A: At 500 I get 87 chunks and drop to 18 out of 30 because small pieces lose surrounding context like Suyya story split. At 1200 I get 36 chunks and stay at 19 out of 30 but long chunks bury phrases and slow precise matching. I kept 800 with 54 chunks because same best score with fewer pieces to search.
Q: The model confidently answers a question your documents do not cover. How did you stop that?
A: Two gates. First, I refuse on raw scores before rescaling: if raw dense best is below 0.25 or raw BM25 best below about 0.3, I return Not found with empty citations and never call the LLM, so refused questions cost zero. Second, survivors must pass a normalized hybrid cutoff 0.85. For example refund policy scores raw dense 0.035, so the first gate already refuses it. Floors were retuned by grid when the corpus shrank to 9 chunks because absolute scores dropped.

# 10. Honest limitations
This now uses hybrid BM25 plus dense embeddings and a grounded LLM with per-sentence cites, and it is live on Railway with Docker slim 478MB. It has a 3-file CC BY-SA corpus (Aryabhata, Brahmagupta, Iron Pillar), no PDF parsing, no vector database, no background jobs, and it reloads and re-chunks on every request so it is slow. It has rate limiting (20/min/IP) and ingest disabled on live, but /ask is open by design for the demo - put behind auth before serious sharing. Refusal is 10/10 tuning and 5/5 held-out; answerable misses are small-corpus ranking limits.

Corpus note: docs/ holds condensed extracts of three Wikipedia articles (Aryabhata, Brahmagupta, Iron pillar of Delhi), CC BY-SA 4.0, source URLs in each header, replacing the previous excerpt. It is not my original writing. If you reuse the repo, replace it with your own documents or a public-domain corpus, and verify you have the right to publish any file you put in docs/.

# 11. How to run it
```bash
git clone https://github.com/Kaustubh070707/production-rag-chatbot.git && cd production-rag-chatbot
cp .env.example .env  # fill in the values listed below
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000/docs
# With Docker:
docker build -t rag:slim .
docker run -p 8000:8000 --env-file .env rag:slim
```
Required environment variables: `NVIDIA_API_KEY`, `NVIDIA_BASE_URL` (default `https://integrate.api.nvidia.com/v1`), `LLM_MODEL` (default `openai/gpt-oss-20b`), `EMBEDDING_MODEL` (default `nvidia/llama-nemotron-embed-vl-1b-v2`)

# 12. References
- NVIDIA API docs for the two models used (embeddings and chat)
- BM25 and hybrid retrieval notes I read while implementing `app/hybrid.py`
- FastAPI docs for the service wiring
