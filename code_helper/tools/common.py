import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
EMBED_MODEL = os.environ["NVIDIA_EMBED_MODEL"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

nvidia = OpenAI(base_url=os.environ["NVIDIA_BASE_URL"],
                api_key=os.environ["NVIDIA_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def embed(texts: list[str], kind: str) -> list[list[float]]:
    """kind is 'passage' for documents, 'query' for questions."""
    resp = nvidia.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": kind, "truncate": "END"},
    )
    return [d.embedding for d in resp.data]