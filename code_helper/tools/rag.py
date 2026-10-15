"""rag.py - semantic code search over the Pinecone index."""

from .common import embed, pc, INDEX_NAME

TOP_K = 5


def rag_search(query: str) -> str:
    """Search the indexed code for chunks relevant to a question.
    Use this when asked where something is implemented or located in the code.
    Args:
        query: The question or functionality to look for, e.g. 'where is the search done'.
    Returns:
        Matching code with file name, line numbers and function name, or a clear message.
    """
    if not isinstance(query, str) or not query.strip():
        return "Please provide a search query."

    try:
        index = pc.Index(INDEX_NAME)
        qvec = embed([query.strip()], "query")[0]
        res = index.query(vector=qvec, top_k=TOP_K, include_metadata=True)
        matches = getattr(res, "matches", None) or res.get("matches", [])
        if not matches:
            return "NO_RESULTS: the search found no matching code. Answer that nothing was found. Do not search again."

        parts = [f"Found {len(matches)} code chunks for: {query.strip()}\n"]
        for i, m in enumerate(matches, 1):
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            score = m["score"] if isinstance(m, dict) else m.score
            parts.append(
                f"[{i}] {md.get('file_path', '?')} "
                f"(lines {md.get('start_line', '?')}-{md.get('end_line', '?')}) "
                f"{md.get('chunk_type', '')}: {md.get('name', '')} (score {float(score):.2f})\n"
                f"{md.get('text', '')}\n"
            )
        return "\n".join(parts)

    except Exception as e:
        return (f"TOOL_ERROR ({type(e).__name__}): {e}. "
                "The search tool failed. Do not call it again. Tell the user the search failed.")