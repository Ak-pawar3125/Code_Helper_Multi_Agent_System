"""reader.py - Stage 1: File discovery and reading."""

import sys
from pathlib import Path
from typing import Sequence
from .config import PROJECT_DIR

EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".idea", ".vscode", "node_modules"}
EXCLUDED_EXTENSIONS = {".ipynb", ".pyc", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz"}


def read_files(
    directory: Path | str | None = None,
    patterns: Sequence[str] = ("*.py",),
    recursive: bool = False,
) -> list[Path]:
    """Discover files to be indexed from the given project directory.

    Args:
        directory: Root path to search for files. Defaults to PROJECT_DIR from config.
        patterns: File globs to match, e.g. ("*.py",) or ("*.py", "*.md").
        recursive: Whether to search recursively in subdirectories.

    Returns:
        A sorted list of valid Path objects.
    """
    base_dir = Path(directory) if directory else PROJECT_DIR
    if not base_dir.exists():
        sys.exit(f"Directory not found: {base_dir}. Please check PROJECT_DIR in .env")

    files: list[Path] = []
    for pattern in patterns:
        matched = base_dir.rglob(pattern) if recursive else base_dir.glob(pattern)
        for p in matched:
            # Skip excluded directories and non-relevant extensions
            if any(part in EXCLUDED_DIRS for part in p.parts):
                continue
            if p.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue
            if p.is_file():
                files.append(p)

    files = sorted(set(files))
    if not files:
        sys.exit(f"No matching files found in {base_dir}. Check PROJECT_DIR or file patterns.")

    return files


def load_file_content(file_path: Path) -> str:
    """Read and return content of a file as UTF-8 string."""
    return file_path.read_text(encoding="utf-8", errors="replace")
