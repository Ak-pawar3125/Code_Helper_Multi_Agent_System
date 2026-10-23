"""index_codebase.py - CLI entrypoint for codebase indexing.

Delegates execution to the modular RAG pipeline in `code_helper.rag_pipeline`.
Stages:
  1. Reader (code_helper.rag_pipeline.reader)
  2. Chunker (code_helper.rag_pipeline.chunker)
  3. Embedder (code_helper.rag_pipeline.embedder)
  4. Store (code_helper.rag_pipeline.store)
  5. Searcher (code_helper.rag_pipeline.searcher)
"""

import argparse
import sys
from pathlib import Path

# Ensure root workspace is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_helper.rag_pipeline import (
    run_indexing,
    read_files,
    chunk_file,
    count_tokens,
    format_embed_input,
    embed_all,
    get_or_create_index,
    clear_index,
    upsert_chunks,
    search_codebase,
)

# Backwards compatibility aliases
embed_input = format_embed_input
get_index = get_or_create_index


def main():
    ap = argparse.ArgumentParser(description="Index codebase into Pinecone using modular RAG pipeline.")
    ap.add_argument("--strategy", choices=["hybrid", "fixed"], default="hybrid", help="Chunking strategy")
    ap.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens per chunk")
    ap.add_argument("--overlap-lines", type=int, default=3, help="Overlapping lines for sliding window")
    ap.add_argument("--namespace", default="", help="Pinecone namespace (leave empty to match default search)")
    ap.add_argument("--no-clean", action="store_false", dest="clean", help="Do not drop existing vectors before upserting")
    ap.add_argument("--dry-run", action="store_true", help="Chunk only without calling embedding API or vector store")
    args = ap.parse_args()

    run_indexing(
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        overlap_lines=args.overlap_lines,
        namespace=args.namespace,
        clean=args.clean,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()