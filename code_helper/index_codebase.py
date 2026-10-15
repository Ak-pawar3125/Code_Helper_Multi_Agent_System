"""index_codebase.py - run once to fill Pinecone.

Steps: 1 read files, 2 split (hybrid chunking), 3 embed, 4 store.
Step 5 (search) lives in tools/rag.py.
"""

import argparse
import ast
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import ServerlessSpec
from code_helper.tools.common import PROJECT_DIR, INDEX_NAME, pc, embed

EMBED_DIM = 2048  # nvidia/nemotron-3-embed-1b

# Token counting: tiktoken if available, otherwise ~4 characters per token.
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# ------------------------- 1. READ ---------------------------------------
def read_files():
    files = sorted(PROJECT_DIR.glob("*.py"))
    if not files:
        sys.exit(f"No .py files found in {PROJECT_DIR}. Check PROJECT_DIR in .env")
    return files


# ----------------------------- 2. SPLIT -----------------------------------
def split_lines(lines, first_line, max_tokens, overlap_lines):
    """Fallback: fill chunks line by line up to max_tokens, repeating a few
    lines between neighbours. Returns (start_line, end_line, text) tuples."""
    pieces, cur, cur_t, start = [], [], 0, first_line
    for line in lines:
        t = count_tokens(line)
        if cur and len(cur) > overlap_lines and cur_t + t > max_tokens:
            pieces.append((start, start + len(cur) - 1, "".join(cur)))
            keep = cur[-overlap_lines:] if overlap_lines else []
            start = start + len(cur) - len(keep)
            cur, cur_t = list(keep), sum(count_tokens(x) for x in keep)
        cur.append(line)
        cur_t += t
    if cur:
        pieces.append((start, start + len(cur) - 1, "".join(cur)))
    return pieces


def chunk_file(path, strategy, max_tokens, overlap_lines):
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    total = len(lines)

    # Decide the blocks: (start_line, end_line, type, name)
    blocks = []
    tree = None
    if strategy == "hybrid":
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
    if tree is None:
        blocks.append((1, total, "fixed-size", "file lines"))
    else:
        defs = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                defs.append((start, node.end_lineno, kind, node.name))
        cursor = 1
        for start, end, kind, name in defs:
            if kind == "module" and count_tokens(text) < 60:
                continue  # skip tiny glue code, e.g. if __name__ == "__main__"
            if start > cursor:
                blocks.append((cursor, start - 1, "module", "module-level code"))
            blocks.append((start, end, kind, name))
            cursor = end + 1
        if cursor <= total:
            blocks.append((cursor, total, "module", "module-level code"))

    chunks = []
    for start, end, kind, name in blocks:
        text = "".join(lines[start - 1:end])
        if not text.strip():
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


def embed_input(c):
    """Text that gets embedded: a small header plus the code."""
    return (f"# file: {c['file_path']} | {c['chunk_type']}: {c['name']} | "
            f"lines {c['start_line']}-{c['end_line']}\n{c['text']}")


# ------------------------ 3. EMBED ----------------------------------------
def embed_all(texts, batch_size=16):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(4):
            try:
                vectors += embed(batch, "passage")
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2 * 2 ** attempt
                print(f"  embedding failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        time.sleep(0.5)
    return vectors


# --------------------------- 4. STORE ------------------------------------
def get_index():
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        print(f"Creating index '{INDEX_NAME}' ({EMBED_DIM} dims, cosine)")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBED_DIM, metric="cosine",
            spec=ServerlessSpec(cloud=os.environ["PINECONE_CLOUD"],
                                region=os.environ["PINECONE_REGION"]))
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        dim = pc.describe_index(INDEX_NAME).dimension
        if dim != EMBED_DIM:
            sys.exit(f"Index '{INDEX_NAME}' has dimension {dim}, expected {EMBED_DIM}. "
                     f"Delete it in the Pinecone dashboard and run again.")
    return pc.Index(INDEX_NAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["hybrid", "fixed"], default="hybrid")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--overlap-lines", type=int, default=3)
    ap.add_argument("--namespace", default="", help="leave empty to match rag.py")
    ap.add_argument("--dry-run", action="store_true", help="chunk only, no API calls")
    args = ap.parse_args()

    files = read_files()
    print("Files:", [f.name for f in files])

    chunks = []
    for f in files:
        c = chunk_file(f, args.strategy, args.max_tokens, args.overlap_lines)
        print(f"  {f.name}: {len(c)} chunks")
        chunks += c
    tokens = [count_tokens(c["text"]) for c in chunks]
    print(f"Total: {len(chunks)} chunks | strategy={args.strategy} "
          f"max_tokens={args.max_tokens} | avg {sum(tokens)//len(tokens)} tokens, "
          f"largest {max(tokens)}")

    if args.dry_run:
        for c, t in zip(chunks, tokens):
            print(f"{c['file_path']:<18} L{c['start_line']}-{c['end_line']:<5} "
                  f"{c['chunk_type']:<10} {c['name']:<28} {t} tokens")
        return

    vectors = embed_all([embed_input(c) for c in chunks])
    print("Vector length:", len(vectors[0]))
    if len(vectors[0]) != EMBED_DIM:
        sys.exit(f"Expected {EMBED_DIM} dimensions. Check NVIDIA_EMBED_MODEL in .env")

    index = get_index()
    try:  # start clean so old chunks from a previous run don't linger
        index.delete(delete_all=True, namespace=args.namespace)
    except Exception:
        pass

    items = [{"id": c["id"], "values": v,
              "metadata": {k: c[k] for k in
                           ("file_path", "start_line", "end_line", "chunk_type", "name", "text")}}
             for c, v in zip(chunks, vectors)]
    for i in range(0, len(items), 100):
        index.upsert(vectors=items[i:i + 100], namespace=args.namespace)

    time.sleep(5)  # stats can lag a few seconds
    print("Index stats:", index.describe_index_stats())
    print(f"Done. Expected about {len(chunks)} vectors.")


if __name__ == "__main__":
    main()