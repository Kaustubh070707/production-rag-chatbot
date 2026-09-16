import json
from pathlib import Path
from app.chunking import chunk_text
from app.vector_retriever import vector_retrieve

def load_chunks():
    chunks = []
    for f in sorted(Path("docs").glob("*.md")) + sorted(Path("docs").glob("*.txt")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(t, size=800, overlap=80))
    return chunks

def run_eval(min_score):
    texts = load_chunks()
    ok = total = ans_ok = ans_tot = ref_ok = ref_tot = 0
    fails = []
    with open("eval/questions.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            total += 1
            hits = vector_retrieve(q["question"], texts, top_k=2)
            top = hits[0][1] if hits else 0.0
            if not q["answerable"]:
                ref_tot += 1
                passed = (not hits or top < min_score)
                ref_ok += 1 if passed else 0
            else:
                ans_tot += 1
                blob = " ".join([h[0] for h in hits])
                passed = (q["must_contain"] in blob) and top >= min_score
                ans_ok += 1 if passed else 0
            ok += 1 if passed else 0
            if not passed:
                fails.append((q["question"], top))
    return ok, total, ans_ok, ans_tot, ref_ok, ref_tot, fails

for ms in [0.3, 0.4, 0.5]:
    ok, total, a_ok, a_tot, r_ok, r_tot, fails = run_eval(ms)
    print(f"MIN={ms}: HIT {ok}/{total} ({ok/total:.0%}) ans={a_ok}/{a_tot} ref={r_ok}/{r_tot}")
    for q, s in fails:
        print(f"  FAIL {s:.3f} {q!r}")
