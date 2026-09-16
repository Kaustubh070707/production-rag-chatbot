import json
from pathlib import Path
from app.chunking import chunk_text
from app.retriever import retrieve

def load_texts():
    texts = []
    for f in sorted(Path("docs").glob("*.md")) + sorted(Path("docs").glob("*.txt")):
        texts.append(f.read_text(encoding="utf-8", errors="ignore"))
    return texts

def run_eval(chunks):
    ok = 0
    total = 0
    ans_ok = 0
    ans_total = 0
    ref_ok = 0
    ref_total = 0
    with open("eval/questions.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            total += 1
            hits = retrieve(q["question"], chunks, top_k=2)
            if not q["answerable"]:
                ref_total += 1
                passed = (not hits or hits[0][1] == 0)
                ref_ok += 1 if passed else 0
            else:
                ans_total += 1
                blob = " ".join([h[0] for h in hits])
                passed = q["must_contain"] in blob
                ans_ok += 1 if passed else 0
            ok += 1 if passed else 0
    return ok, total, ans_ok, ans_total, ref_ok, ref_total

for size, overlap in [(500, 50), (800, 80), (1200, 100)]:
    chunks = []
    for t in load_texts():
        chunks.extend(chunk_text(t, size=size, overlap=overlap))
    ok, total, a_ok, a_tot, r_ok, r_tot = run_eval(chunks)
    print(f"{size}/{overlap}: chunks={len(chunks)} HIT {ok}/{total} ({ok/total:.0%}) ans={a_ok}/{a_tot} ref={r_ok}/{r_tot}")
