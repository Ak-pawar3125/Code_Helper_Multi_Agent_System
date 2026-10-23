"""searcher.py - Stage 5: Semantic search and precision retrieval over Pinecone."""

from typing import Any
from .config import pc_client, INDEX_NAME, TOP_K
from .embedder import embed_batch


def search_codebase(
    query: str,
    top_k: int = TOP_K,
    namespace: str = "",
    file_filter: str = "",
) -> str:
    """Search indexed code for chunks relevant to a query.

    Args:
        query: User question or natural language description of functionality.
        top_k: Maximum number of matches to return.
        namespace: Pinecone namespace to search.
        file_filter: Optional filter to restrict matches to a specific file.

    Returns:
        Formatted markdown string containing matches or an informative status message.
    """
    if not isinstance(query, str) or not query.strip():
        return "Please provide a valid search query."

    if pc_client is None:
        return "Pinecone client not configured. Set PINECONE_API_KEY in .env."

    try:
        index = pc_client.Index(INDEX_NAME)

        # Generate query embedding with input_type='query'
        query_vectors = embed_batch([query.strip()], kind="query")
        if not query_vectors:
            return "Failed to generate query embedding."
        qvec = query_vectors[0]

        # Build query parameters
        filter_dict = {}
        if file_filter:
            filter_dict["file_path"] = {"$eq": file_filter}

        res = index.query(
            vector=qvec,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
        )

        matches = getattr(res, "matches", None) or res.get("matches", [])
        if not matches:
            return "NO_RESULTS: the search found no matching code. Answer that nothing was found. Do not search again."

        parts = [f"Found {len(matches)} code chunks for: {query.strip()}\n"]
        for i, m in enumerate(matches, 1):
            md: dict[str, Any] = m["metadata"] if isinstance(m, dict) else m.metadata
            score: float = m["score"] if isinstance(m, dict) else m.score

            file_path = md.get("file_path", "unknown")
            start_l = md.get("start_line", "?")
            end_l = md.get("end_line", "?")
            chunk_type = md.get("chunk_type", "code")
            name = md.get("name", "")
            code_text = md.get("text", "").strip()

            parts.append(
                f"[{i}] {file_path} (lines {start_l}-{end_l}) "
                f"{chunk_type}: {name} (relevance score: {float(score):.2f})\n"
                f"```python\n{code_text}\n```\n"
            )

        return "\n".join(parts)

    except Exception as e:
        return (
            f"TOOL_ERROR ({type(e).__name__}): {e}. "
            "The search tool failed. Do not call it again. Tell the user the search failed."
        )
