import json
from app.main import load_documents
from app.retriever import retrieve

items = load_documents()
texts = [t for _, t in items]
ok = 0
total = 0
with open("eval/questions.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        total += 1
        hits = retrieve(q["question"], texts, top_k=2)
        if not q["answerable"]:
            passed = (not hits or hits[0][1] == 0)
            print(f"{q['question']!r} -> refusal expected, top={hits[0][1] if hits else 0} {'PASS' if passed else 'FAIL'}")
        else:
            blob = " ".join([h[0] for h in hits])
            passed = q["must_contain"] in blob
            print(f"{q['question']!r} -> must_contain={q['must_contain']!r} {'PASS' if passed else 'FAIL'} top={hits[0][1] if hits else 0}")
        ok += 1 if passed else 0
print(f"HIT {ok}/{total}")
