"""read_file.py — Read and return the complete contents of a source file in the project directory."""

from ._helpers import _get_project_py_files

from ..common import PROJECT_DIR


def read_file(file_name: str) -> str:
    """Read and return the complete contents of a source file in the project directory.
    Use this when asked to explain or read a file, e.g. 'explain vector_store.py'.
    Args: file_name: The name of the file to read (e.g. 'vector_store.py' or 'main.py').
    Returns: The full text content of the file, or an error message if not found.
    """
    if not file_name or not isinstance(file_name, str):
        return "Invalid file name: name cannot be empty."
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        return f"Security error: Invalid file name '{file_name}'. Path separators and '..' are not allowed."
    clean_name = file_name.strip()
    target_path = PROJECT_DIR / clean_name
    if not target_path.exists() or not target_path.is_file():
        available = [p.name for p in _get_project_py_files()]
        avail_str = ", ".join(available) if available else "none"
        return f"File '{clean_name}' not found. Available project files: {avail_str}."
    try:
        return target_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading '{clean_name}': {e}."
