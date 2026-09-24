# RAG Chatbot — Hybrid Retrieval with Eval Harness

> Demo RAG service that answers strictly from your own documents — with citations, and refusal when the answer isn't there. Built to show retrieval quality with numbers, not just a chatbot UI.

A FastAPI service with hybrid BM25 + dense retrieval and grounded LLM answers. Demo corpus is `docs/notes.md` (591 lines, study excerpt — not my original writing; replace it with your own documents). Live demo needs an NVIDIA key on the server side, not in the browser.

## Demo

- **Live:** https://production-rag-chatbot-production.up.railway.app (try `/health`, then `/docs` → `POST /ask`)
- **Demo video:** https://youtu.be/i_eClL5WwuQ (2 min: live health, cited answer, refusal)
- **Repo:** https://github.com/Kaustubh070707/production-rag-chatbot

Try in Swagger: open `/docs` → `POST /ask`

```json
{
  "query": "Who was Aryabhata?",
  "top_k": 2
}
```

Real response from the live service (untrimmed, scores are hybrid normalized scores — 1.0 is the per-query rescaled top, see note below):

```json
{
  "query": "Who was Aryabhata?",
  "answer": "Aryabhata I (476 CE) was the first astronomer who tackled the problems of new astronomy. [notes.md]\nHe invented a system of expressing numbers with consonants and vowels. [notes.md]\nHe laid the foundations of scientific Indian astronomy in 499 CE. [notes.md]\nHe taught astronomy to pupils who included Pandurangasvamin, Latadeva, and Nihsanka. [notes.md]\nHe was from Kusumapura (Pataliputra or Patna). [notes.md]",
  "citations": [
    {
      "source": "notes.md",
      "score": 1.0,
      "chunk": "Aryabhata I (476 CE) was the first astronomer who tackled the problems of new\nastronomy. He invented a system of expressing numbers with the help of\nconsonants and vowels, based again on the decimal place value principle. The\nsystem was used by Bhaskara I (574 CE) and Aryabhata II (950 CE), and app"
    },
    {
      "source": "notes.md",
      "score": 0.881,
      "chunk": "lendar than in the\nGregorian calendar.\nCultural Developments\nIndia's first satellite Aryabhata and the lunar crater Aryabhata were named\nto honour this great Indian scientist. Further, the Aryabhatta Research\nInstitute of Observational Sciences (ARIES) as a centre for research and\ntraining in astrop"
    }
  ],
  "llm_latency_ms": 1800,
  "llm_tokens": 310
}
```

Scores: `1.0` is the best chunk after per-query min-max rescaling, so a weak question can still show 1.0 for its top — raw scores (dense cosine, BM25) are the refusal signal. See “How refusal works” below.

Unanswerable questions return `Not found in your documents.` with empty citations and no LLM call — tested as **refused 10/10 unanswerable test questions on the tuning set**, not “no hallucinations” in general.

## Architecture

```
[notes.md] -> [chunk 800/80 -> 54 pieces] -> [BM25 + dense vectors]
[query] -> [hybrid retrieve top-2 + raw-score refusal gates] -> [LLM grounded answer + cites]
```

- Chunking 800 chars / 80 overlap (compared 500/1200, kept 800: same accuracy, fewer pieces).
- Hybrid retrieval: BM25 keyword scores plus dense cosine, min-max combined, with raw-score floors that refuse weak matches before ranking.
- Grounded generation: strict context-only prompt, per-sentence `[source]` cites, extractive fallback if the LLM fails. Refused questions never reach the LLM.

## Results (same 30 tuning questions throughout, plus a held-out check)

Tuning set = 30 questions I wrote and tuned cutoffs/stopwords on (20 answerable with exact phrases verified in notes, 10 unanswerable). Held-out = 12 new questions written after freezing all thresholds. Reproduce with commands in the last column.

| Stage | Hit-rate | Notes |
|---|---|---|
| Word-match 800/80 tuning | 19/30 (63%) | First real baseline. Two misses split across chunk boundaries. `python eval/run_eval.py` |
| BM25 expanded-stopwords MIN=1.0 tuning | 25/30 (83%) | Rare Vitasta now outweighs common India. `python eval/run_eval_bm25.py` |
| Dense cosine MIN=0.3 tuning | 26/30 (87%) | Refusals 10/10, lost 4 precise-keyword Qs. `python eval/run_eval_vector.py` |
| Hybrid raw-gate OR (dense 0.2 / bm25 1.0) + MIN 0.85 tuning | **29/30 (97%) tuning** — 19/20 answerable, 10/10 refusals; only Brahmagupta split-phrase miss | Current best on tuning set. `python eval/run_eval_hybrid.py` |
| Held-out 12 questions (never tuned on) | 12/12 (100%) — 7/7 answerable, 5/5 refusals | New phrasings and trick refusals written after freezing thresholds. `python eval/run_eval_heldout.py` |
| Grounded answers faithfulness (5 samples, strict prompt) | 5/5 grounded, 5/5 clean per-sentence cites | No preamble leaks after strict prompt. `python eval/run_faithfulness.py` |

Full 16-row lab notebook (all chunk sizes, cutoffs, intermediate hybrids) is in `docs/EVAL_LOG.md` — the table above is the story.

**Reproducibility**

```bash
# retrieval (tuning set)
python eval/run_eval.py
python eval/run_eval_bm25.py
python eval/run_eval_hybrid.py   # current best 29/30
python eval/run_eval_heldout.py  # held-out 10/12

# image sizes
docker build -f Dockerfile.naive -t rag:naive . && docker images rag:naive
docker build -t rag:slim . && docker images rag:slim
# expect 1.9GB naive -> 478MB slim (~75% smaller), both `curl /health` ok

# live smoke
curl https://production-rag-chatbot-production.up.railway.app/health
```

## How refusal works

Two gates, in order: raw dense best `<0.2` or raw BM25 best `<1.0` → refuse immediately (cheap, absolute, saves LLM cost). Survivors are rescaled per-query and must pass normalized hybrid `>=0.85`. Per-query rescaling makes ranking fair but erases absolute strength — the raw gate keeps the no-match signal.

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
- `POST /ingest` → file upload (currently MD supported; PDF/DOCX and background worker are next — see Limitations)

## Limitations (honest) and Next

This is a demo, not production:

- Single-file markdown corpus (`docs/notes.md` excerpt) — replace it with your own; in-memory list + cached vectors, no vector database, reloads on every request.
- No auth or rate limiting on `/ask` and `/ingest` — the live Railway URL lets anyone hit it and spend the NVIDIA key. Put it behind auth or a rate limit before sharing broadly.
- No CI on this repo yet — the sibling `cicd-pipeline` repo holds the lint/test/scan/push/deploy pipeline that will ship this service.
- Tests are the eval harness, not unit tests; ingest is MD-only; no re-rank yet (last miss is a chunk-boundary split that re-rank might fix), no repeat-query cache.

Next: auth on live, persistent pgvector, PDF ingest, repeat-query cache, cross-encoder re-rank.

## Repo layout

```
app/            FastAPI service (chunking, BM25, dense, hybrid, LLM)
docs/           corpus (your documents go here) + EVAL_LOG.md full table
eval/           30-question tuning set (questions.jsonl) + held-out set + runners
Dockerfile      multi-stage slim (~478MB)
Dockerfile.naive  unoptimized baseline (~1.9GB)
SKILL.md        engineering log — decisions, numbers, failures (kept for transparency; interview answers are rehearsed from measured numbers)
DEBUG_LOG.md    full code-level debug history (kept outside docs/ so it never pollutes retrieval)
```

## Corpus note

`docs/notes.md` is a study excerpt used as a demo dataset. It is not my original writing. If you reuse the repo, replace it with your own documents or a public-domain corpus, and verify you have the right to publish any file you put in `docs/`.
