import json
from pathlib import Path
from app.chunking import chunk_text
from app.hybrid import hybrid_retrieve_with_raw

EVAL_FILE = Path("eval/questions_heldout.jsonl")

def load_chunks():
    chunks = []
    for f in sorted(Path("docs").glob("*.md")) + sorted(Path("docs").glob("*.txt")):
        t = f.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(t, size=800, overlap=80))
    return chunks

def run_eval(alpha=0.5, dense_floor=0.2, bm25_floor=1.0, hybrid_min=0.85):
    texts = load_chunks()
    ok = total = ans_ok = ans_tot = ref_ok = ref_tot = 0
    fails = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            total += 1
            ranked, rd, rb = hybrid_retrieve_with_raw(q["question"], texts, top_k=2, alpha=alpha)
            top = ranked[0][1] if ranked else 0.0
            if not q["answerable"]:
                ref_tot += 1
                passed = (rd < dense_floor or rb < bm25_floor or top < hybrid_min)
                ref_ok += 1 if passed else 0
            else:
                ans_tot += 1
                blob = " ".join([c for c, _ in ranked])
                in_top = q["must_contain"] in blob
                gate_refused = (rd < dense_floor or rb < bm25_floor)
                passed = in_top and not gate_refused and top >= hybrid_min
                ans_ok += 1 if passed else 0
            ok += 1 if passed else 0
            if not passed:
                fails.append((q["question"], round(rd,3), round(rb,3), round(top,3)))
    return ok, total, ans_ok, ans_tot, ref_ok, ref_tot, fails

if __name__ == "__main__":
    ok, total, a_ok, a_tot, r_ok, r_tot, fails = run_eval()
    print(f"HIT {ok}/{total} ({ok/total:.0%}) ans={a_ok}/{a_tot} ref={r_ok}/{r_tot}")
    for f in fails:
        print(f"  FAIL raw_dense={f[1]} raw_bm25={f[2]} hybrid={f[3]} {f[0]!r}")
    print("Run: python eval/run_eval_heldout.py (held-out, never tuned on — thresholds frozen from tuning set)")
