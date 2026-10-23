"""pipeline.py - Orchestrator coordinating all RAG indexing stages."""

import time
from pathlib import Path
from typing import Any

from .reader import read_files
from .chunker import chunk_file, format_embed_input, count_tokens
from .embedder import embed_all
from .store import get_or_create_index, clear_index, upsert_chunks, get_index_stats


def run_indexing(
    directory: Path | str | None = None,
    strategy: str = "hybrid",
    max_tokens: int = 300,
    overlap_lines: int = 3,
    namespace: str = "",
    clean: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the complete end-to-end RAG indexing pipeline.

    Stages:
      1. Read: Discover source files in target directory.
      2. Split: Parse and chunk files into semantic code blocks.
      3. Embed: Generate vector embeddings via NVIDIA NIM API.
      4. Store: Upsert chunks and vectors into Pinecone serverless index.

    Returns:
        Summary dict containing statistics about files, chunks, and vectors.
    """
    print("=" * 60)
    print("  Starting RAG Indexing Pipeline")
    print("=" * 60)

    # --- Stage 1: Read Files ---
    print("\n[Stage 1/4] Discovering files...")
    files = read_files(directory=directory, patterns=("*.py",))
    print(f"  Found {len(files)} files: {[f.name for f in files]}")

    # --- Stage 2: Chunk / Split ---
    print(f"\n[Stage 2/4] Chunking files (strategy={strategy}, max_tokens={max_tokens}, overlap={overlap_lines})...")
    chunks: list[dict[str, Any]] = []
    for f in files:
        file_chunks = chunk_file(f, strategy=strategy, max_tokens=max_tokens, overlap_lines=overlap_lines)
        print(f"  {f.name}: {len(file_chunks)} chunks")
        chunks.extend(file_chunks)

    if not chunks:
        print("  No chunks generated. Terminating pipeline.")
        return {"files": len(files), "chunks": 0, "vectors": 0}

    tokens = [count_tokens(c["text"]) for c in chunks]
    avg_tokens = sum(tokens) // len(tokens) if tokens else 0
    max_chunk_tokens = max(tokens) if tokens else 0
    print(f"  Total chunks: {len(chunks)} | Avg tokens: {avg_tokens} | Max tokens: {max_chunk_tokens}")

    if dry_run:
        print("\n[Dry Run Mode] Skipping embedding and vector store operations.")
        for c, t in zip(chunks, tokens):
            print(f"  {c['file_path']:<20} L{c['start_line']}-{c['end_line']:<5} {c['chunk_type']:<10} {c['name']:<25} ({t} tokens)")
        return {
            "files": len(files),
            "chunks": len(chunks),
            "vectors": 0,
            "dry_run": True,
        }

    # --- Stage 3: Embed Texts ---
    print(f"\n[Stage 3/4] Generating embeddings for {len(chunks)} chunks...")
    embed_inputs = [format_embed_input(c) for c in chunks]
    vectors = embed_all(embed_inputs, kind="passage", batch_size=16)
    print(f"  Generated {len(vectors)} vectors (dimension: {len(vectors[0])})")

    # --- Stage 4: Store in Pinecone ---
    print("\n[Stage 4/4] Connecting to Pinecone vector store...")
    index = get_or_create_index()

    if clean:
        print(f"  Cleaning existing vectors in namespace '{namespace}'...")
        clear_index(index, namespace=namespace)

    print(f"  Upserting {len(chunks)} vectors to Pinecone...")
    upserted_count = upsert_chunks(index, chunks, vectors, namespace=namespace, batch_size=100)
    print(f"  Successfully upserted {upserted_count} vectors.")

    # Brief delay for Pinecone stats sync
    time.sleep(3)
    stats = get_index_stats(index)
    print(f"  Index Stats: {stats}")

    print("\n" + "=" * 60)
    print(f"  RAG Pipeline Completed! Indexed {upserted_count} chunks.")
    print("=" * 60)

    return {
        "files": len(files),
        "chunks": len(chunks),
        "vectors": upserted_count,
        "dry_run": False,
    }
