import json
from pathlib import Path
from app.chunking import chunk_text
from app.hybrid import hybrid_retrieve_with_raw
texts=[]
for f in sorted(Path("docs").glob("*.md"))+sorted(Path("docs").glob("*.txt")):
 texts.extend(chunk_text(f.read_text(encoding="utf-8",errors="ignore"),size=800,overlap=80))
def ev(questions, alpha, dfl, bfl, hmin):
 ok=tot=aa=at=ra=rt=0
 for line in open(questions,encoding="utf-8"):
  line=line.strip()
  if not line: continue
  q=json.loads(line); tot+=1
  ranked,rd,rb=hybrid_retrieve_with_raw(q["question"],texts,top_k=2,alpha=alpha)
  top=ranked[0][1] if ranked else 0
  if not q["answerable"]:
   rt+=1; p=(rd<dfl or rb<bfl or top<hmin); ra+=1 if p else 0
  else:
   at+=1; blob=" ".join([c for c,_ in ranked]); p=(q["must_contain"] in blob) and not(rd<dfl or rb<bfl) and top>=hmin; aa+=1 if p else 0
  ok+=1 if p else 0
 return ok,tot,aa,at,ra,rt
for dfl,bfl in [(0.15,0.3),(0.2,0.3),(0.25,0.3),(0.2,0.5),(0.25,0.5),(0.15,1.0)]:
 ok,tot,aa,at,ra,rt=ev("eval/questions.jsonl",0.5,dfl,bfl,0.85)
 print(f"dfl={dfl} bfl={bfl}: HIT {ok}/{tot} ans={aa}/{at} ref={ra}/{rt}")
