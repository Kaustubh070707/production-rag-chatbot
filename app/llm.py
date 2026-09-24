import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You answer ONLY from the Context below. "
    "Rules: "
    "1. Write short sentences, one fact per sentence. "
    "2. End EVERY sentence with the exact bracketed label shown before its chunk, e.g. [notes.md]. "
    "3. If the Context lacks the answer, reply with exactly this and nothing else: Not found in your documents. "
    "4. Never use outside knowledge. "
    "5. Return ONLY the final answer. No preamble, no reasoning, no planning, no meta-commentary. "
    "Example - Context: [notes.md] Aryabhata I (476 CE) was the first astronomer who tackled new astronomy. "
    "Question: Who was Aryabhata? "
    "Correct: Aryabhata I (476 CE) was the first astronomer who tackled new astronomy. [notes.md] "
    "Wrong (never do this): We need to answer... / Thus... / sentences without the label / facts not in Context."
)



def _client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set (add it to .env)")
    return OpenAI(api_key=api_key,base_url=NVIDIA_BASE_URL)

def _build_user_message(query:str,cited:list[tuple[str,str]]) -> str:
    # Header label matches the citation label the model must emit, e.g. [notes.md].
    # Example output for cited=[("notes.md","Aryabhata I (476 CE) was...")]:
    #   Question: Who was Aryabhata?
    #   Context:
    #   [notes.md]
    #   Aryabhata I (476 CE) was...
    parts = [f"Question: {query}", "Context:"]
    for source,chunk in cited:
        parts.append(f"[{source}]")
        parts.append(chunk)
    return "\n".join(parts)

def generate_answer(query:str,cited:list[tuple[str,str]],metrics:dict | None = None)->str:
    client = _client()
    response = client.chat.completions.create(
        model = os.getenv("LLM_MODEL",DEFAULT_LLM_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role":"user","content": _build_user_message(query,cited)}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=False,
    )

    if metrics is not None and response.usage is not None:
        metrics["prompt_tokens"] = response.usage.prompt_tokens
        metrics["completion_tokens"] = response.usage.completion_tokens

    content = response.choices[0].message.content or ""
    return content.strip()
