# RAG Chatbot — Hybrid Retrieval with Eval Harness

> Demo RAG service that answers strictly from your own documents — with citations, and refusal when the answer isn't there. Built to show retrieval quality with numbers, not just a chatbot UI.

A FastAPI service with hybrid BM25 + dense retrieval and grounded LLM answers. Demo corpus is `docs/` — three Wikipedia extracts (Aryabhata, Brahmagupta, Iron Pillar of Delhi), CC BY-SA 4.0 with source URLs in each file header. Replace them with your own documents anytime.

## Demo

- **Live:** https://production-rag-chatbot-production.up.railway.app (try `/health`, then `/docs` → `POST /ask`)
- **Demo video:** https://youtu.be/i_eClL5WwuQ (2 min: live health, cited answer, refusal)

Try in Swagger: open `/docs` → `POST /ask`

```json
{
  "query": "Who was Aryabhata?",
  "top_k": 2
}
```

Real response from the live service after the CC corpus swap (warm, no edits — chunks truncated to 300 characters by the API at `app/main.py:58`, scores hybrid normalized where 1.0 is the per-query rescaled top):

```json
{
  "query": "Who was Aryabhata?",
  "answer": "Aryabhata I (476-550 CE) was the first major mathematician-astronomer from the classical age of Indian mathematics and astronomy. [aryabhata.md] He authored the Aryabhatiya in 499 CE. [aryabhata.md] He called himself a native of Kusumapura or Pataliputra. [aryabhata.md] He set up an observatory at the Sun temple in Taregana, Bihar. [aryabhata.md] His calendar calculations influenced the Jalali calendar. [aryabhata.md] India's first satellite and a lunar crater were named after him. [aryabhata.md] The Aryabhatta Research Institute of Observational Sciences is named in his honour. [aryabhata.md] His successor Bhaskara I wrote commentaries on his system. [aryabhata.md]",
  "citations": [
    {
      "source": "aryabhata.md",
      "score": 1.0,
      "chunk": "# Aryabhata — Ancient Indian Astronomer\n\nSource: Wikipedia article \"Aryabhata\" (CC BY-SA 4.0, https://en.wikipedia.org/wiki/Aryabhata, retrieved 2026). Condensed extract for demo retrieval corpus.\n\nAryabhata I (476-550 CE) was the first of the major mathematician-astronomers from the classical age o"
    },
    {
      "source": "aryabhata.md",
      "score": 0.8400214587548946,
      "chunk": "s calendar calculations fed the Jalali calendar introduced in 1073 CE by astronomers including Omar Khayyam, still the basis of national calendars in Iran and Afghanistan.\n\nIndia's first satellite Aryabhata and the lunar crater Aryabhata are named in his honour, as is the Aryabhatta Research Institu"
    }
  ],
  "llm_latency_ms": 13619.6,
  "llm_tokens": 1187
}
```

Chunks above are truncated to 300 characters by the API (`chunk[:300]` in `app/main.py:58`); the full chunk is longer. Scores are normalized per-query, so `1.0` is the best for that query — raw dense/BM25 scores are the refusal signal. See “How refusal works” below.

Unanswerable questions return `Not found in your documents.` with empty citations and no LLM call — tested as **refused 10/10 unanswerable test questions on the tuning set**, not “no hallucinations” in general. Open the frontend at `/` or use `/docs`.

## Architecture

```
[docs/: 3 Wikipedia files] -> [chunk 800/80 -> 9 pieces] -> [BM25 + dense vectors]
[query] -> [hybrid retrieve top-2 + raw-score refusal gates] -> [LLM grounded answer + cites]
```

- Chunking 800 chars / 80 overlap (compared 500/1200 on the previous corpus, kept 800: same accuracy, fewer pieces).
- Hybrid retrieval: BM25 keyword scores plus dense cosine, min-max combined, with raw-score floors (dense 0.25 / bm25 0.3, retuned for this corpus) that refuse weak matches before ranking.
- Grounded generation: strict context-only prompt, per-sentence `[source]` cites, extractive fallback if the LLM fails. Refused questions never reach the LLM.

## Results (same 30 tuning questions throughout, plus a held-out check)

Tuning set = 30 questions I wrote and tuned cutoffs/stopwords on (20 answerable with exact phrases verified in corpus, 10 unanswerable). Held-out = 12 new questions written after freezing all thresholds. Corpus is now 3 CC BY-SA Wikipedia extracts (previous 591-line excerpt replaced — see Corpus note). Reproduce with commands in the last column.

| Stage | Hit-rate | Notes |
|---|---|---|
| Word-match 800/80 tuning | 19/30 (63%, previous corpus) | First real baseline on old corpus, kept for history. |
| BM25 expanded-stopwords tuning | 25/30 (83%, previous corpus) | Rare-terms outweigh common ones. Old corpus numbers. |
| Hybrid raw-gate OR (dense 0.25 / bm25 0.3) + MIN 0.85 tuning | **27/30 (90%) tuning** — 17/20 answerable, 10/10 refusals | Current corpus. Floors retuned (9-chunk corpus scores lower absolutely). `python eval/run_eval_hybrid.py` + `eval/grid_floors.py` |
| Held-out 12 questions (never tuned on) | 10/12 (83%) — 5/7 answerable, 5/5 refusals | New phrasings on new corpus, thresholds frozen. `python eval/run_eval_heldout.py` |
| Grounded answers faithfulness (5 samples, strict prompt) | 5/5 grounded, 5/5 clean per-sentence cites on new corpus | No preamble leaks; per-file cites ([aryabhata.md] etc.); avg ~806 tokens/Q. `python eval/run_faithfulness.py` |

Full 16-row lab notebook (all chunk sizes, cutoffs, intermediate hybrids) is in `eval/EVAL_LOG.md` — the table above is the story.

**Reproducibility**

```bash
# retrieval (tuning set) — needs no key for BM25; dense/hybrid need NVIDIA_API_KEY in .env
python eval/run_eval.py
python eval/run_eval_bm25.py
python eval/run_eval_hybrid.py   # current best 29/30 tuning
python eval/run_eval_heldout.py  # held-out 12/12 — thresholds frozen before this file was added (see git log 77862bc -> 0979bc9)

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

- CC BY-SA corpus (`docs/aryabhata.md`, `brahmagupta.md`, `iron-pillar.md`, condensed Wikipedia extracts with source URLs) — replace with your own; in-memory list + cached vectors, no vector database, reloads on every request.
- Rate limiting (20/min/IP) on `/ask`, ingest disabled on live (`503`) — `/ask` still open by design for the demo; put behind auth before any serious sharing.
- Pipeline wired in-repo (`.github/workflows/pipeline.yml` + `k8s/`, lint/test/scan/push/kind deploy) — same SHA flow as the `cicd-pipeline` sibling.
- Tests are 4 unit tests plus the eval harness; ingest is MD-only stub; misses are small-corpus ranking limits (9 chunks), not calibration bugs.

Next: persistent pgvector, PDF ingest worker, cross-encoder re-rank, repeat-query cache.

## Repo layout

```
app/            FastAPI service (chunking, BM25, dense, hybrid, LLM)
static/         minimal frontend (ask UI, disabled ingest notice)
docs/           CC BY-SA corpus (3 Wikipedia extracts, attributed) — replace with your own
eval/           30-question tuning set (questions.jsonl) + held-out set + runners + EVAL_LOG.md full table
tests/          unit tests (chunking, health, refusal gate)
Dockerfile      multi-stage slim (~468MB)
Dockerfile.naive  unoptimized baseline (~1.9GB)
SKILL.md        engineering log — decisions, numbers, failures
DEBUG_LOG.md    full code-level debug history (kept outside docs/ so it never pollutes retrieval)
```

## Corpus note

`docs/` holds condensed extracts of three Wikipedia articles (Aryabhata, Brahmagupta, Iron pillar of Delhi), CC BY-SA 4.0, with source URLs in each file header, replacing the previous textbook excerpt. If you reuse the repo, replace them with your own documents or another CC-licensed source, and verify you have the right to publish any file you put in `docs/`.
