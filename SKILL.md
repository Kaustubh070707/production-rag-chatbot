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
<Two sentences. Explain it to a non-technical friend, then to an engineer.>
Non-technical: A chatbot that answers only from my documents with citations.
Engineer: FastAPI RAG service with ingestion, hybrid BM25 + vector retrieval, cross-encoder re-rank, pgvector store.

# 2. Problem it solves
<Why would anyone run this? If the honest answer is "it was a tutorial", change the project until there is a real answer.>
Answers questions strictly from supplied corpus (PDF/DOCX/HTML/MD) with refusal when context insufficient.

# 3. Architecture
<Paste an ASCII or image diagram. Every box must be something you can explain.>
```
[Docs] -> [Ingest + Chunk + Embed] -> [pgvector + BM25]
[User Q] -> [Hybrid Retrieve top-k] -> [Re-rank] -> [LLM + strict prompt] -> [Answer + citations]
```
Components:
- ingest -> parse + chunk with overlap -> chose chunk size after eval, not before
- retriever -> hybrid vector + BM25 -> vector alone misses keywords
- api -> FastAPI + background worker for parsing/embedding -> uploads stay fast
- eval -> 30 labelled Qs, hit-rate + faithfulness -> proves quality

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
