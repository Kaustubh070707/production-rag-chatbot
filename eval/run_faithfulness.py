import json
from app.main import load_documents
from app import hybrid
from app.llm import generate_answer

QS = [
    ("Who was Aryabhata?", "Aryabhata"),
    ("What did Suyya build?", "Vitasta"),
    ("What is special about Delhi iron pillar?", "phosphorus"),
    ("Who built Kaveri Anicut?", "Kaveri"),
    ("What is Nagarjuna work on alchemy?", "Rasaratnakara"),
]

pairs = load_documents()
texts = [c for _, c in pairs]
src = {c: s for s, c in pairs}
tot_p = tot_c = 0
faith = 0
for q, must in QS:
    ranked, rd, rb = hybrid.hybrid_retrieve_with_raw(q, texts, top_k=2, alpha=0.5)
    if rd < 0.2 or rb < 1.0 or not ranked or ranked[0][1] < 0.85:
        print(f"Q {q!r}: REFUSED (gates {rd:.2f}/{rb:.2f})")
        continue
    cited = [(src[c], c) for c, _ in ranked[:2]]
    m = {}
    a = generate_answer(q, cited, m)
    pt, ct = m.get("prompt_tokens", 0), m.get("completion_tokens", 0)
    tot_p += pt
    tot_c += ct
    cites = a.count("[notes.md]")
    grounded = must.lower() in a.lower()
    faith += 1 if grounded else 0
    tail_ok = not a.rstrip().endswith(("Aryabh", "Competi", "institut"))
    pre = a.lstrip().startswith(("We need", "The user asks", "Thus"))
    print(f"Q {q!r}: pt={pt} ct={ct} cites={cites} must={grounded} clean_end={tail_ok} preamble={pre}")
    print("  " + a[:220].encode("ascii", "replace").decode() + "...")
print(f"FAITH {faith}/{len(QS)} TOKENS prompt={tot_p} completion={tot_c} total={tot_p+tot_c} avg/Q={(tot_p+tot_c)/len(QS):.0f}")
