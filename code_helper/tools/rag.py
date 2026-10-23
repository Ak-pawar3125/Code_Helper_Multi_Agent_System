"""rag.py - semantic code search over the Pinecone index."""

from ..rag_pipeline.searcher import search_codebase
from ..rag_pipeline.config import TOP_K


def rag_search(query: str) -> str:
    """Search the indexed code for chunks relevant to a question.
    Use this when asked where something is implemented or located in the code.
    Args:
        query: The question or functionality to look for, e.g. 'where is the search done'.
    Returns:
        Matching code with file name, line numbers and function name, or a clear message.
    """
    return search_codebase(query=query, top_k=TOP_K)