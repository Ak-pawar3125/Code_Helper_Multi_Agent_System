"""_helpers.py — Shared private helpers for code_tools."""

import ast
from pathlib import Path

from ..common import PROJECT_DIR


def _get_project_py_files() -> list[Path]:
    """Return all top-level .py files in PROJECT_DIR sorted by filename."""
    if not PROJECT_DIR.exists() or not PROJECT_DIR.is_dir():
        return []
    return sorted([p for p in PROJECT_DIR.glob("*.py") if p.is_file()])


def _validate_py_file(file_name: str) -> tuple[Path | None, str | None]:
    """Validate that file_name is a safe, existing top-level .py file."""
    if not file_name or not isinstance(file_name, str):
        return None, "Invalid file name: name cannot be empty."
    if ".." in file_name or "/" in file_name or "\\" in file_name:
        return None, f"Security error: Invalid file name '{file_name}'. Path separators and '..' are not allowed."
    clean_name = file_name.strip()
    if not clean_name.endswith(".py"):
        return None, f"Invalid file type: '{clean_name}' is not a Python (.py) file."
    target_path = PROJECT_DIR / clean_name
    if not target_path.exists() or not target_path.is_file():
        available = [p.name for p in _get_project_py_files()]
        avail_str = ", ".join(available) if available else "none"
        return None, f"File '{clean_name}' not found. Available project files: {avail_str}."
    return target_path, None


def _parse_ast(path: Path) -> tuple[ast.Module | None, str | None]:
    """Safely parse a Python file into an AST module, returning an error message on failure."""
    try:
        content = path.read_text(encoding="utf-8")
        return ast.parse(content, filename=path.name), None
    except SyntaxError as e:
        return None, f"Syntax error in '{path.name}' at line {e.lineno}: {e.msg}."
    except Exception as e:
        return None, f"Error reading '{path.name}': {e}."


def _get_file_imports(tree: ast.Module, project_stems: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Extract project and external imports from an AST tree."""
    project_imports: dict[str, list[str]] = {}
    external_imports: dict[str, list[str]] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                root_pkg = mod_name.split(".")[0]
                target_dict = project_imports if root_pkg in project_stems else external_imports
                if mod_name not in target_dict:
                    target_dict[mod_name] = []
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_name = node.module
                root_pkg = mod_name.split(".")[0]
                target_dict = project_imports if root_pkg in project_stems else external_imports
                imported_names = [alias.name for alias in node.names]
                if mod_name not in target_dict:
                    target_dict[mod_name] = []
                target_dict[mod_name].extend(imported_names)

    return project_imports, external_imports


def _clean_summary(doc: str, filename: str = "") -> str:
    """Extract and clean the first meaningful summary line from a module docstring."""
    if not doc:
        return "Python source file"
    lines = [line.strip() for line in doc.strip().splitlines() if line.strip()]
    if not lines:
        return "Python source file"
    summary = lines[0]
    if filename:
        for prefix in (f"{filename} —", f"{filename} -", f"{filename}:"):
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
                break
    return summary or "Python source file"
