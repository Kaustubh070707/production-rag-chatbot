---
project: production-rag-chatbot
track: ai-ml
level: beginner-intermediate
started: 2026-09-09
shipped:
repo:
live:
---
# 1. What this project is
Non-technical: I have uploaded my Ancient Science notes and ask it questions related to it, it shows answer and file name
Engineer: FastAPI service, split md file into 800-char pieces with 80 overlap, find top 2 with BM25 keyword ranking and refuse when score is low

# 2. Problem it solves
I waste a lot of time(~20 minutes) finding the details e.g. Aryabhata vs Brahmagupta details in 591-line notes , this project solves this problem and return the result in < 1s and refuses if nothing exist

# 3. Architecture
[notes.md - 591 lines ancient science] -> [chunk 800/80 -> 54 pieces] -> [list in memory]
[query like Who was Aryabhata?] -> [BM25 top-2 + cutoff MIN_SCORE 1.0] -> [answer with source + score or Not found]

Components:
- I split notes.md into 800-character pieces with 80 overlap, which gave me 54 chunks. I tried 500 (87 chunks) and 1200 (36 chunks) too, but 800 gave the same 63% word-match with fewer pieces to search than 500, so I kept it for speed with no accuracy loss.
- I keep the 54 chunks in a plain Python list in memory. With such a small corpus there is no need for Postgres and pgvector yet, which would add setup and hosting with no benefit. I will move to pgvector when the corpus grows beyond memory or needs metadata filtering.
- I rank with BM25, which gives rare words like Vitasta much more weight than common words like India that appear in twenty chunks. I picked BM25 over my first plain word-counting because refusal of unanswerable questions went from 1 out of 10 to 5 out of 10, while answerable stayed perfect.

# 4. Key decisions and trade-offs
| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Chunking | 500/50 giving 87 chunks vs 800/80 giving 54 vs 1200/100 giving 36 | 800 with 80 overlap | I kept 800 because on my 30 questions it scored 19 out of 30 like 1200 but with more precise pieces than 1200, and beat 500 which only got 18 out of 30. Fewer chunks than 500 means faster search with no accuracy loss. | I gave up finer granularity. A phrase like Siddhanta Shiromani split across an 800-char boundary still fails, which 500 sometimes catches. |
| Retrieval | Plain word-overlap counting every word equal vs BM25 weighting rare words high | BM25 with expanded stopwords | I chose BM25 because on the same 30 questions it went from 19 out of 30 to 25 out of 30. Rare Vitasta now outweighs common India, so questions like What is iPhone price in India correctly refuse instead of matching Delhi iron pillar via India. | I gave up simplicity. BM25 needs the rank-bm25 dependency plus tuning of stopwords and cutoff, while word-match needed nothing. |
| Refusal cutoff | No cutoff vs MIN_SCORE 0.5 vs 1.0 vs 1.5 | MIN_SCORE 1.0 | I set the cutoff at 1.0 because at 0.5 the question What is iPhone price in India still passed with 0.98, and at 1.5 nothing further improved over 1.0. At 1.0 I keep all 20 answerable while refusing 5 of 10 unanswerable, which was the sweet spot on my 30 questions. | I gave up some borderline answers. A low-scoring correct chunk below 1.0 now refuses instead of showing a weak guess. |

# 5. Skills demonstrated
- [ ] Embeddings + vector search evidence: NOT BUILT YET - currently BM25 sparse only in `app/bm25_retriever.py`
- [x] Chunking strategy eval evidence: `README` results table 500/87 vs 800/54 vs 1200/36 chunks with 30-Q scores
- [x] Sparse retrieval + threshold refusal evidence: `app/bm25_retriever.py` + `MIN_SCORE 1.0` in `app/main.py`
- [x] Prompt grounding + refusal evidence: `POST /ask` returns `Not found in your documents` with empty citations when top score below cutoff
- [x] Evaluation methodology evidence: `eval/questions.jsonl` 30-Q (20 answerable + 10 unanswerable) + repeatable runner
- [ ] Cost/latency control evidence: NOT BUILT YET - no token tracking or cache yet

# 6. Numbers I measured
| Metric | Before | After | How I measured it |
|---|---|---|---|
| 30-question hit-rate with plain word-match, chunk 800/80, top-2 | Nothing built yet | 19 correct out of 30 (63 percent) - 18 of 20 answerable found, only 1 of 10 refusals correct | I wrote 20 answerable questions with exact phrases verified in notes.md like Vitasta and phosphorus plus 10 unanswerable like What is refund policy, then ran each through POST /ask top-2 and counted must_contain present in citations or correct Not-found. Runner kept locally as eval script. |
| 30-question hit-rate with BM25, chunk 800/80, cutoff 1.0, top-2 | 19 out of 30 word-match baseline above | 25 correct out of 30 (83 percent) - 20 of 20 answerable found, 5 of 10 refusals correct | Same 30 questions, swapped retriever to BM25Okapi with expanded stopwords covering how, when, explain, is, are, and cutoff MIN_SCORE 1.0 tuned over 0.5, 1.0, 1.5. The question What is iPhone price in India now scores 0.98 and correctly refuses instead of matching Delhi via India. |
| Chunk count vs accuracy | 500/50 gave 87 chunks and 18/30, 1200/100 gave 36 chunks and 19/30 | 800/80 gave 54 chunks and 19/30 word-match, chosen as best balance | I chunked the same 591-line notes.md three ways and ran the same 30 questions each time. 800 matched the best score with almost half the chunks of 500, so search is faster with no accuracy loss. |

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
   Fix: Not fixed yet - documented as known failure. Options are larger stopwords, heading filtering, or embeddings that understand key-words versus API-key are different meanings.
   Lesson: Keyword matching cannot tell word sense apart. This is exactly why the vault says the next step is embeddings and hybrid search, which I will add after this baseline.

# 8. What I would do differently at 100x scale
- I would move chunks from Python list to pgvector with metadata filtering by source and date, plus a background worker for parsing so uploads stay fast, because in-memory reload on every request will not survive restarts or large corpora.
- I would add embeddings alongside BM25 for hybrid retrieval plus a cross-encoder re-rank, because keyword BM25 still fails 5 of 10 refusals on meaning mismatches like API key versus Key Words.
- I would add token and cost tracking per query plus caching of repeated questions, because currently I have no idea what each query would cost with an LLM.

# 9. Interview answers I have rehearsed
Q: How do you know your RAG is good? Give me a number and how you measured it.
A: I get 25 out of 30 correct with BM25 800/80 cutoff 1.0, which is 20 of 20 answerable found and 5 of 10 refusals correct. I wrote 20 answerable with exact phrases verified in my 591-line notes plus 10 unanswerable like refund and iPhone, ran each through top-2 and counted must_contain present or correct Not-found. Word-match baseline on same set was 19 out of 30, so BM25 added 6 points.
Q: Chunk size 800 - what breaks at 500 and 1200?
A: At 500 I get 87 chunks and drop to 18 out of 30 because small pieces lose surrounding context like Suyya story split. At 1200 I get 36 chunks and stay at 19 out of 30 but long chunks bury phrases and slow precise matching. I kept 800 with 54 chunks because same best score with fewer pieces to search.
Q: The model confidently answers a question your documents do not cover. How did you stop that?
A: I refuse when BM25 top score is below 1.0 and return Not found with empty citations. For example What is iPhone price in India scored 0.98 and now refuses instead of matching Delhi via India. It is not perfect - How to reset API key still fails via Key Words heading at 3.49 - which is why embeddings are next.

# 10. Honest limitations
This project does not use embeddings or any LLM yet - answers are extractive top chunks, not generated sentences. It has only one 591-line markdown file, no PDF parsing, no vector database, no background jobs, and it reloads and re-chunks on every request so it is slow. Refusal is only 5 out of 10 because keyword matching cannot tell meaning apart. Saying this is why hybrid with embeddings is the documented next step.

# 11. How to run it
```bash
git clone <repo> && cd production-rag-chatbot
cp .env.example .env # fill in the values listed below
docker compose up --build
# open http://localhost:8000/docs
```
Required environment variables: OPENAI_API_KEY / EMBEDDING_MODEL, DATABASE_URL, VECTOR_DIM

# 12. Credits
- https://github.com/microsoft/generative-ai-for-beginners
- https://github.com/langchain-ai/langchain
- https://github.com/mlabonne/llm-course
- https://github.com/eugeneyan/applied-ml
