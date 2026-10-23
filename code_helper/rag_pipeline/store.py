"""store.py - Stage 4: Pinecone vector index management and upserts."""

import sys
import time
from typing import Any, Sequence
from pinecone import ServerlessSpec
from .config import pc_client, INDEX_NAME, EMBED_DIM, PINECONE_CLOUD, PINECONE_REGION


def get_or_create_index(index_name: str = INDEX_NAME, dimension: int = EMBED_DIM):
    """Retrieve existing Pinecone index or create it if it does not exist."""
    if pc_client is None:
        sys.exit("Pinecone client is not initialized. Please set PINECONE_API_KEY.")

    existing_indexes = [i.name for i in pc_client.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' ({dimension} dims, metric=cosine)...")
        pc_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        while not pc_client.describe_index(index_name).status["ready"]:
            print("  Waiting for Pinecone index to become ready...")
            time.sleep(1)
    else:
        current_dim = pc_client.describe_index(index_name).dimension
        if current_dim != dimension:
            sys.exit(
                f"Index '{index_name}' has dimension {current_dim}, expected {dimension}. "
                "Delete it in the Pinecone dashboard and re-run."
            )

    return pc_client.Index(index_name)


def clear_index(index, namespace: str = "") -> None:
    """Clear all vectors from the specified namespace to ensure fresh indexing."""
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Cleared existing vectors in namespace '{namespace}'.")
    except Exception as e:
        # Ignore if index is already empty or namespace doesn't exist yet
        pass


def upsert_chunks(
    index,
    chunks: Sequence[dict[str, Any]],
    vectors: Sequence[list[float]],
    namespace: str = "",
    batch_size: int = 100,
) -> int:
    """Format and upsert chunks with their vector embeddings into Pinecone.

    Args:
        index: Pinecone Index instance.
        chunks: List of chunk metadata dictionaries.
        vectors: List of embedding vectors matching chunks.
        namespace: Pinecone namespace.
        batch_size: Number of vectors per upsert request.

    Returns:
        Total number of vectors upserted.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors.")

    items = []
    for c, v in zip(chunks, vectors):
        items.append({
            "id": c["id"],
            "values": v,
            "metadata": {
                "file_path": c.get("file_path", ""),
                "start_line": c.get("start_line", 0),
                "end_line": c.get("end_line", 0),
                "chunk_type": c.get("chunk_type", ""),
                "name": c.get("name", ""),
                "text": c.get("text", ""),
            },
        })

    total_upserted = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    return total_upserted


def get_index_stats(index):
    """Retrieve index statistics."""
    return index.describe_index_stats()
