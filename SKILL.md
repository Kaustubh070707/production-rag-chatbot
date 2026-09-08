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
Engineer: FastAPI service, split md file into 500-char pieces , find top 3 by word match

# 2. Problem it solves
I waste a lot of time(~20 minutes) finding the details e.g. Aryabhata vs Brahmagupta details in 591-line notes , this project solves this problem and return the result in < 1s and refuses if nothing exist

# 3. Architecture
[notes.md] -> [chunk 500/50] -> [list in memory]
[query] -> [word-match top-3] -> [answer + citations]

Components:
- chunk 500/50 -> splits notes.md to 500-char pieces -> chose 500 not 100 (loses context) or 2000 (dilutes match), will prove with eval
- list in memory -> holds ~54 chunks in Python list -> chose memory not Postgres/pgvector yet because small + zero setup, move when grows
- word-match top-3 -> scores by common words ignoring who/was/the -> chose this not embeddings because no key needed for baseline

# 4. Key decisions and trade-offs
| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Vector DB | pgvector vs Qdrant vs Chroma | pgvector | Postgres already needed, filters + vectors together | Managed-scale ease of Qdrant |
| Chunking | 500/overlap vs 100 vs 2000 | TBD after eval | Measure hit-rate first |  |
| Retrieval | pure vector vs hybrid + re-rank | hybrid + cross-encoder | Better recall + precision | Extra latency/cost |

# 5. Skills demonstrated
- [ ] Embeddings + vector search evidence: `app/retriever.py`
- [ ] Chunking strategy eval evidence: `README` results table
- [ ] Hybrid retrieval + re-ranking evidence: `app/retriever.py`
- [ ] Prompt grounding + refusal evidence: `app/prompts.py`
- [ ] Evaluation methodology evidence: `eval/questions.jsonl` + hit-rate script
- [ ] Cost/latency control evidence: token tracking + cache

# 6. Numbers I measured
| Metric | Before | After | How I measured it |
|---|---|---|---|
| retrieval hit-rate top-5 | 61% baseline TBD | 89% target | 30-Q labelled set |
| per-query token cost | TBD | TBD | token counter per request |
| upload p95 | TBD | <300ms | BullMQ/Celery offload |

# 7. Things that broke and how I fixed them
1. Symptom:
   Cause:
   Fix:
   Lesson:

# 8. What I would do differently at 100x scale
- TBD: separate ingest workers, cached embeddings, sharded vector index
- TBD:
- TBD:

# 9. Interview answers I have rehearsed
Q: How do you know your RAG is good? Give me a number and how you measured it.
A:
Q: Chunk size 500 - what breaks at 100 and 2000?
A:
Q: Model answers what docs don't cover - how did you stop that?
A:

# 10. Honest limitations
<What this does NOT do. Saying this out loud builds trust.>

# 11. How to run it
```bash
git clone <repo> && cd production-rag-chatbot
cp .env.example .env # fill in the values listed below
docker compose up --build
# open http://localhost:8000/docs
```
Required environment variables: OPENAI_API_KEY / EMBEDDING_MODEL, DATABASE_URL, VECTOR_DIM

# 12. Credits
<Any tutorial/repo/article you learned from. Copying is fine; uncredited copying is not.>
- https://github.com/microsoft/generative-ai-for-beginners
- https://github.com/langchain-ai/langchain
- https://github.com/mlabonne/llm-course
- https://github.com/eugeneyan/applied-ml
