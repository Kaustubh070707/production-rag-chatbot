# Debug Log - production-rag-chatbot (chronological)

From first file to BM25 work. Full code-level history. Kept OUT of `docs/` on purpose - anything in `docs/` gets chunked and retrieved, which would contaminate the 30-Q eval.

---

## 1. `chunking.py` - `List[String]` after the colon
- **Symptom:** `chunk_text('hello')` -> `NameError: name 'List' is not defined`; function returned nothing useful.
- **Cause:** `def chunk_text(text,size=500,overlap=50):List[String]` was all on **one line**. Python parsed `List[String]` as the *function body* (a one-line expression body), not a return annotation. Also `List`/`String` aren't Python built-ins (`list`/`str` are).
- **Fix:** `def chunk_text(text, size=500, overlap=50) -> list[str]:` with the body indented on the next line.
- **Lesson:** `def f(): expr` on one line is *valid* Python - it becomes a one-line body. Type annotations need `->` on the same line + a newline for the body.

## 2. `chunking.py` - overlap only applied to first pair + tail dropped
- **Symptom:** For `(len=1200, size=100, overlap=20)`, only chunks 1-2 overlapped (20 chars); all later pairs had **0 overlap**; and when `len % size == 0`, the last `overlap` chars were silently never chunked.
- **Cause:** `for i in range(0,len,size): i -= overlap` - reassigning the loop variable inside a `for` over an iterator has **no effect on the next iteration**. The window step stayed `size`, not `size-overlap`, and the end drifted short of the text tail.
- **Fix:** slide with `while start < len: ...; start += size - overlap`, or equivalently `for k in range(n_chunks): start = k*(size-overlap)`.
- **Lesson:** In `for i in range(...)`, you cannot control iteration by mutating `i` - the iterator hands you the next value. Use `while` or compute values *from* the index.

## 3. `chunking.py` - short texts dropped
- **Symptom:** `chunk_text('abc')` -> `[]`; the whole document vanished. 102/3000 randomized coverage tests failed, all with `chunks=0`.
- **Cause:** `n_chunks = (len - overlap + step - 1) // step` yields `0` when `len <= overlap` - negative numerator floored to 0.
- **Fix:** `if len(text) <= size: return [text]`.
- **Lesson:** Guard small inputs; formulas have domains. "If len < size, return the whole thing" is the correct contract.

## 4. `chunking.py` - wrong error message
- **Symptom:** `ValueError: overlap must be in [0,100]): got 100` - `])` typo and the range shown (`[0,100]`) contradicts the check (`overlap < size`).
- **Fix:** `f"overlap must be in [0, {size}); got {overlap}"`.
- **Lesson:** Error messages should mirror the actual validation range, and `]` vs `)` is user-visible truth.

## 5. `retriever.py` - `KeyError` on the first word
- **Symptom:** `retrieve('hello world', [...])` -> `KeyError: 'hello'` on every call.
- **Cause:** `query_dict[temp] += 1` on a key that doesn't exist yet - `+=` reads the key first.
- **Fix:** `text_dict[word] = text_dict.get(word, 0) + 1` (or `defaultdict(int)`/`Counter`).
- **Lesson:** Counting words with `+=` needs an existing key; `.get(k, 0) + 1` is the standard idiom.

## 6. `retriever.py` - `temp.to_lower`
- **Symptom:** `AttributeError: 'str' object has no attribute 'to_lower'`.
- **Cause:** Typo - Python's method is `.lower()`.
- **Fix:** `.lower()`.
- **Lesson:** Python method names are exact; `to_lower` is C#/other-language style.

## 7. `retriever.py` - dead branch -> tokenizer always returned `{}`
- **Symptom:** `tokenizer('hello world')` -> `{}`, all scores `0.0`.
- **Cause:** `elif ch != " ":` was nested *inside* `if ch == " ":` - inside that block `ch` is always a space, so the accumulation branch was **dead code**. Non-space characters were never collected.
- **Fix:** Restructured: if char is `a-z0-9` -> accumulate; else (space/punct/newline) -> finalize word + reset.
- **Lesson:** Check where your `if/elif` blocks actually live - indentation can silently kill branches. Also: simplify tokenization to one rule ("keep a-z0-9, else split").

## 8. `retriever.py` - missing spec features
- **Symptom:** No lowercase, no `a-z0-9` filter, stopwords and `len<=2` words kept, trailing word unfiltered, `get_score` a stub (`0.0`), no sorting, `top_k` unused.
- **Cause:** The char-loop attempt handled none of the spec's filters and the scorer was a placeholder.
- **Fix:** Full rewrite - one `tokenize` walk applying *all* filters consistently (mid-word *and* trailing word), real `get_score = common / query_words`, `sort(reverse=True)`, `[:top_k]`.
- **Lesson:** Write the spec as a checklist (lowercase / filter / split / stopwords / len / score / sort / top_k / empty-safe) and verify each line explicitly.

## 9. `retriever.py` - division by zero (latent)
- **Symptom:** Empty query -> `query_words = 0` -> `ZeroDivisionError` once real scoring existed.
- **Fix:** `if not query_map or not chunk_map: return 0.0` - empty either side scores 0.0 by spec.
- **Lesson:** Guard denominators *before* implementing the formula, not after the crash.

## 10. `main.py` - `/ask` was missing half the spec
- **Symptom:** Loaded a hardcoded single file (not `docs/*.md`), lost `(source, chunk)` pairs, had no "No documents loaded", no refusal on `score==0`, citations used the wrong key (`chunks`) with no `source`/`[:300]`, answer was raw text.
- **Cause:** Spec return shape not followed.
- **Fix:** Added `load_documents()` (glob + chunk 800/80 + `(source, chunk)`), empty-doc response, refusal response, `citations=[{source, score, chunk[:300]}]`, `answer=f"Top match from {source}: {chunk[:500]}"`.
- **Lesson:** When a spec lists a return shape, build and test the exact shape (`query`+`answer`+`citations`) - every field and key name.

## 11. `bm25_retriever.py` - `BM250kapi`
- **Symptom:** Pylance: `"BM250kapi" is unknown import symbol`.
- **Cause:** Typed a **zero** (`BM25**0**kapi`) instead of the **letter O** (`BM25**O**kapi`).
- **Fix:** Correct spelling; confirmed the real class: `class BM25Okapi(BM25)` in `rank_bm25.py:78`.
- **Lesson:** O vs 0 in identifiers is invisible to the eye but real to Python. `Ctrl+Click` the symbol to check it resolves.

## 12. `bm25_retriever.py` - dict tokenizer (duplicates lost)
- **Symptom:** Copied the old `retriever` tokenizer returning `dict[str,int]` - `"Suyya Suyya Vitasta"` -> `{'suyya':2,'vitasta':1}`.
- **Cause:** BM25 needs **term frequency**, so repeated words must survive as a list.
- **Fix:** `tokenize` returns `list[str]` (append duplicates): `['suyya','suyya','vitasta']`.
- **Lesson:** Know your algorithm's data requirements - TF models need lists, not sets/dicts.

## 13. `bm25_retriever.py` - Pylance `float64` vs `float`
- **Symptom:** `list[tuple[str, float64]]` is not assignable to `list[tuple[str, float]]`.
- **Cause:** `BM25Okapi.get_scores()` returns a **numpy `float64` ndarray**, not Python floats.
- **Fix:** `scores: list[float] = [float(s) for s in bm25.get_scores(qtokens)]`.
- **Lesson:** numpy returns `float64`; cast to plain `float` to satisfy types *and* keep FastAPI/JSON serialization clean.

## 14. `main.py` - still wired to the old retriever
- **Symptom:** `/ask` called `retriever.retrieve(...)` and refused with `== 0.0`.
- **Cause:** The swap was planned but never applied; BM25 scores are unbounded (`2.41` for "Aryabhata"), so `== 0.0` is the wrong refusal test.
- **Fix:** `bm25_retriever.bm25_retrieve(...)` + `MIN_SCORE = 1.0` + `results[0][1] < MIN_SCORE`.
- **Lesson:** When you swap a component, audit every downstream assumption (thresholds, return shapes, types).

## 15. `requirements.txt` - `rank_bm25` missing
- **Symptom:** Imports worked locally (installed in `.venv`) but a fresh `pip install -r requirements.txt` would fail.
- **Cause:** Installed but never declared.
- **Fix:** Add `rank_bm25>=0.2.2`.
- **Lesson:** Installed != declared. Any package you import must be in `requirements.txt`.

## 16. `bm25_retriever.py` - stopword set too small
- **Symptom:** Question words (`how/when/where/what/why/explain/tell/is/are/do/did/can...`) pollute BM25 query tokens.
- **Cause:** Initial set of 14 missed question stems.
- **Fix:** Extended `STOPWORDS` from 14 -> 29 words, lifting 30-Q from 21/30 to 25/30.
- **Lesson:** Stopwords should be tuned to your domain - question stems are worthless for scoring.

---

### Recurring themes (the meta-lessons)
1. **Read your own indentation** - silent dead code caused the two weirdest bugs (#2, #7).
2. **Don't mutate loop variables in `for`** - compute from the index instead (#2).
3. **Guard edge cases first** - empty/short inputs (#3, #9) crash or silently lose data.
4. **Match the spec's contract exactly** - keys, thresholds, slices (#10, #14).
5. **Type-check numeric boundaries** - numpy `float64` vs `float` (#13).
6. **Verify with real runs** - each fix was proven with live execution before wiring.
