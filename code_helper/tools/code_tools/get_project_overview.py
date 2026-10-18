"""get_project_overview.py — Provide a complete project overview, file summaries, reading order, and how to run it."""

import ast

from ._helpers import _get_project_py_files, _parse_ast, _clean_summary

from ..common import PROJECT_DIR


def get_project_overview() -> str:
    """Provide a complete project overview, file summaries, reading order, and how to run it.
    Use this when asked for a project overview, where to start, or how to onboard.
    Args: None.
    Returns: A structured string covering project purpose, file summaries, reading order, and how to run it.
    """
    try:
        py_files = _get_project_py_files()
        if not py_files:
            return f"TOOL_ERROR (FileNotFound): No Python files found in project directory: {PROJECT_DIR}."

        # 1. Module summaries from docstrings
        file_summaries: list[str] = []
        for file_path in py_files:
            tree, _ = _parse_ast(file_path)
            doc = ast.get_docstring(tree) if tree else None
            if doc:
                summary = _clean_summary(doc, file_path.name)
                file_summaries.append(f"- {file_path.name}: {summary}")
            else:
                file_summaries.append(f"- {file_path.name}: Python source file")

        # 2. Extract main.py docstring / usage instructions
        main_path = PROJECT_DIR / "main.py"
        usage_info = ""
        if main_path.exists() and main_path.is_file():
            tree, _ = _parse_ast(main_path)
            main_doc = ast.get_docstring(tree) if tree else None
            if main_doc:
                usage_info = main_doc.strip()
            else:
                usage_info = "Run `python main.py` to start the demo."
        else:
            usage_info = "main.py not found in project directory."

        # 3. Recommended reading order
        reading_order = [
            "Recommended Reading Order for New Developers:",
            "1. config.py — Configuration, environment credentials, and data models.",
            "2. indexer.py — Knowledge document loading, AST chunking, and embedding pipeline.",
            "3. vector_store.py — Vector database management (Pinecone with local JSON fallback).",
            "4. query_pipeline.py — Live query embedding, top-k retrieval, and answer synthesis.",
            "5. main.py — End-to-end interactive CLI demonstration connecting all stages.",
        ]

        # 4. What the project does
        purpose = (
            "What the project does:\n"
            "This project implements a complete Naive RAG (Retrieval-Augmented Generation) pipeline.\n"
            "- It loads knowledge documents, chunks them, embeds them via NVIDIA NIM, and stores them in Pinecone.\n"
            "- When queried, it retrieves the most relevant chunks and generates grounded, factual answers."
        )

        separator = "-" * 60
        return (
            f"{purpose}\n\n"
            + "\n".join(reading_order) + "\n\n"
            + "Project Files:\n"
            + "\n".join(file_summaries) + "\n\n"
            + f"How to run the project:\n{separator}\n"
            + usage_info + "\n"
            + f"{separator}"
        )
    except Exception as e:
        return f"TOOL_ERROR ({type(e).__name__}): Failed to generate project overview: {e}"
