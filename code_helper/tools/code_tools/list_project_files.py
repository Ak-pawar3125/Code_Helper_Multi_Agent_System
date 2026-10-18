"""list_project_files.py — List all Python files in the project."""

from ._helpers import _get_project_py_files
from ..common import PROJECT_DIR


def list_project_files() -> str:
    """List all Python files in the project with their line counts.
    Use this when asked which files exist, e.g. 'What files are in the project?'.
    Args: None.
    Returns: A newline-separated string of each Python file and its line count.
    """
    py_files = _get_project_py_files()
    if not py_files:
        return f"No Python files found in project directory: {PROJECT_DIR}."

    lines_out = []
    for file_path in py_files:
        try:
            line_count = len(file_path.read_text(encoding="utf-8", errors="replace").splitlines())
            lines_out.append(f"{file_path.name} ({line_count} lines)")
        except Exception as e:
            lines_out.append(f"{file_path.name} (error reading lines: {e})")

    return "\n".join(lines_out)
