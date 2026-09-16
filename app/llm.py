from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You answer ONLY from the Context below. "
    "End EVERY sentence with the exact source label in square brackets shown in the Context, e.g. [notes.md]. "
    "If the Context lacks the answer, reply exactly: Not found in your documents. "
    "Never use outside knowledge. "
    "Return ONLY the final answer. No preamble, no reasoning, no meta-commentary."
)



def _client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set (add it to .env)")
    return OpenAI(api_key=api_key,base_url=NVIDIA_BASE_URL)

def _build_user_message(query:str,cited:list[tuple[str,str]]) -> str:
    parts = [f"Question: {query}", "Context:"]
    for source,chunk in cited:
        parts.append(f"[source {source}]")
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
