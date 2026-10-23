"""chunker.py - Stage 2: Token counting and AST hybrid code chunking."""

import ast
from pathlib import Path
from typing import Any

# Token counting: tiktoken if available, otherwise ~4 characters per token.
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def split_lines(
    lines: list[str],
    first_line: int,
    max_tokens: int = 300,
    overlap_lines: int = 3,
) -> list[tuple[int, int, str]]:
    """Fill chunks line-by-line up to max_tokens with line overlap between neighbors.

    Returns:
        List of (start_line, end_line, chunk_text) tuples.
    """
    pieces: list[tuple[int, int, str]] = []
    cur: list[str] = []
    cur_t: int = 0
    start: int = first_line

    for line in lines:
        t = count_tokens(line)
        if cur and len(cur) > overlap_lines and (cur_t + t > max_tokens):
            pieces.append((start, start + len(cur) - 1, "".join(cur)))
            keep = cur[-overlap_lines:] if overlap_lines else []
            start = start + len(cur) - len(keep)
            cur = list(keep)
            cur_t = sum(count_tokens(x) for x in keep)
        cur.append(line)
        cur_t += t

    if cur:
        pieces.append((start, start + len(cur) - 1, "".join(cur)))

    return pieces


def chunk_file(
    path: Path,
    strategy: str = "hybrid",
    max_tokens: int = 300,
    overlap_lines: int = 3,
) -> list[dict[str, Any]]:
    """Chunk a file into semantic pieces (functions, classes, module code).

    Args:
        path: Path to the source file.
        strategy: 'hybrid' (AST parsing + line fallback) or 'fixed' (line-by-line only).
        max_tokens: Maximum token count per chunk.
        overlap_lines: Overlapping line count for sliding window chunks.

    Returns:
        List of chunk dicts containing metadata and text.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        return []

    # Decide block ranges: (start_line, end_line, type, name)
    blocks: list[tuple[int, int, str, str]] = []
    tree = None

    if strategy == "hybrid" and path.suffix.lower() == ".py":
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None

    if tree is None:
        blocks.append((1, total, "fixed-size", "file lines"))
    else:
        defs: list[tuple[int, int, str, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                defs.append((start, getattr(node, "end_lineno", node.lineno), kind, node.name))

        cursor = 1
        for start, end, kind, name in defs:
            if start > cursor:
                blocks.append((cursor, start - 1, "module", "module-level code"))
            blocks.append((start, end, kind, name))
            cursor = end + 1

        if cursor <= total:
            blocks.append((cursor, total, "module", "module-level code"))

    chunks: list[dict[str, Any]] = []
    for start, end, kind, name in blocks:
        text = "".join(lines[start - 1:end])
        if not text.strip():
            continue

        # Skip trivial glue code if desired (e.g. very small module blocks)
        if kind == "module" and count_tokens(text) < 15 and len(lines) > 20:
            continue

        if count_tokens(text) <= max_tokens:
            pieces = [(start, end, text)]
        else:
            pieces = split_lines(lines[start - 1:end], start, max_tokens, overlap_lines)

        for s, e, t in pieces:
            chunks.append({
                "id": f"{path.name}:{s}-{e}",
                "file_path": path.name,
                "start_line": s,
                "end_line": e,
                "chunk_type": kind,
                "name": name,
                "text": t,
            })

    return chunks


def format_embed_input(chunk: dict[str, Any]) -> str:
    """Format chunk text with a metadata header for semantic embedding."""
    return (
        f"# file: {chunk['file_path']} | {chunk['chunk_type']}: {chunk['name']} | "
        f"lines {chunk['start_line']}-{chunk['end_line']}\n{chunk['text']}"
    )
