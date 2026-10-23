"""embedder.py - Stage 3: Text embedding generation with NVIDIA NIM."""

import sys
import time
from typing import Sequence
from .config import nvidia_client, EMBED_MODEL, EMBED_DIM


def embed_batch(texts: Sequence[str], kind: str = "passage") -> list[list[float]]:
    """Embed a single batch of texts.

    Args:
        texts: List of strings to embed.
        kind: 'passage' for indexing documents, 'query' for search queries.

    Returns:
        List of embedding float vectors.
    """
    if nvidia_client is None:
        sys.exit("NVIDIA client is not initialized. Please set NVIDIA_API_KEY and NVIDIA_BASE_URL.")

    resp = nvidia_client.embeddings.create(
        model=EMBED_MODEL,
        input=list(texts),
        extra_body={"input_type": kind, "truncate": "END"},
    )
    return [d.embedding for d in resp.data]


def embed_all(
    texts: Sequence[str],
    kind: str = "passage",
    batch_size: int = 16,
    max_retries: int = 4,
    sleep_between_batches: float = 0.5,
) -> list[list[float]]:
    """Embed all texts in batches with exponential backoff on transient errors.

    Args:
        texts: Full sequence of strings to embed.
        kind: 'passage' or 'query'.
        batch_size: Chunk batch size sent to the API.
        max_retries: Retry attempts per batch.
        sleep_between_batches: Rate-limit cooling pause in seconds.

    Returns:
        List of embedding vectors corresponding to each input text.
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        success = False

        for attempt in range(max_retries):
            try:
                batch_vectors = embed_batch(batch, kind=kind)
                vectors.extend(batch_vectors)
                success = True
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}") from e
                wait = 2 * (2 ** attempt)
                print(f"  [Embedder] Batch {i // batch_size + 1} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)

        if not success:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")

        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    # Validate output vector dimensions
    if vectors and len(vectors[0]) != EMBED_DIM:
        raise ValueError(f"Expected {EMBED_DIM} embedding dimensions, but got {len(vectors[0])}.")

    return vectors
