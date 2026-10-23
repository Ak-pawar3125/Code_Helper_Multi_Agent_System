"""config.py - Shared configuration and client initialization for RAG pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env in code_helper or project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", str(Path(__file__).resolve().parent.parent.parent / "Sample_Project")))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "code-helper")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
EMBED_DIM = 2048  # nvidia/nemotron-3-embed-1b dimension
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
TOP_K = 5

_nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
_pinecone_api_key = os.getenv("PINECONE_API_KEY", "")

# Initialize clients lazily or globally
nvidia_client = OpenAI(base_url=_nvidia_base_url, api_key=_nvidia_api_key) if _nvidia_api_key else None
pc_client = Pinecone(api_key=_pinecone_api_key) if _pinecone_api_key else None
