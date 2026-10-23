"""RAG Pipeline Package for Code Helper.

Modular pipeline stages:
- Stage 1: reader - Discover and load project files
- Stage 2: chunker - Token counting and AST hybrid code splitting
- Stage 3: embedder - NVIDIA NIM vector embedding generation
- Stage 4: store - Pinecone vector index management and batch upserts
- Stage 5: searcher - Precision semantic search and retrieval
"""

from .reader import read_files, load_file_content
from .chunker import count_tokens, split_lines, chunk_file, format_embed_input
from .embedder import embed_batch, embed_all
from .store import get_or_create_index, clear_index, upsert_chunks, get_index_stats
from .searcher import search_codebase
from .pipeline import run_indexing
from .config import PROJECT_DIR, INDEX_NAME, EMBED_DIM, EMBED_MODEL, TOP_K

__all__ = [
    # Config
    "PROJECT_DIR",
    "INDEX_NAME",
    "EMBED_DIM",
    "EMBED_MODEL",
    "TOP_K",
    # Stage 1: Reader
    "read_files",
    "load_file_content",
    # Stage 2: Chunker
    "count_tokens",
    "split_lines",
    "chunk_file",
    "format_embed_input",
    # Stage 3: Embedder
    "embed_batch",
    "embed_all",
    # Stage 4: Store
    "get_or_create_index",
    "clear_index",
    "upsert_chunks",
    "get_index_stats",
    # Stage 5: Searcher
    "search_codebase",
    # Pipeline Orchestrator
    "run_indexing",
]
