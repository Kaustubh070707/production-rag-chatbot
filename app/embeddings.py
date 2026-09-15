from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL","https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL:str = os.getenv("EMBEDDING_MODEL","nvidia/llama-nemotron-embed-vl-1b-v2")

def _client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set (add it to .env)")
    return OpenAI(api_key=api_key,base_url=NVIDIA_BASE_URL)

def embed_texts(texts:list[str],input_type:str)->list[list[float]]:
    if not texts:
        return []
    if input_type not in ("query","passage"):
        raise ValueError(f"input_type must be 'query' or 'passage'; got {input_type!r}")

    client = _client()
    response = client.embeddings.create(
        input=texts,
        model=DEFAULT_MODEL,
        encoding_format="float",
        extra_body={"modality": ["text"]*len(texts), "input_type": input_type, "truncate": "NONE"}
    )
    return [[float(x) for x in emb.embedding] for emb in response.data]