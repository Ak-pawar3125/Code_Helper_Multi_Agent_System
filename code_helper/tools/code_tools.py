"""code_tools.py — AST-based code analysis tools for Code Helper agents."""

import ast
from pathlib import Path
from typing import Any

from .common import PROJECT_DIR


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


def list_functions(file_name: str) -> str:
    """List all classes, methods, and functions in a file with lines and arguments.
    Use this when asked what a file defines, e.g. 'What functions are in indexer.py?'.
    Args: file_name: The Python file to inspect, e.g. 'indexer.py'.
    Returns: A formatted string listing classes, methods, and functions.
    """
    path_obj, error = _validate_py_file(file_name)
    if error or not path_obj:
        return error or "Unknown validation error."

    tree, parse_error = _parse_ast(path_obj)
    if parse_error or not tree:
        return parse_error or "Unable to parse file."

    elements: list[str] = [f"{path_obj.name}:"]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            elements.append(f"  class {node.name} (lines {start}-{end}):")
            methods_found = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods_found = True
                    m_start = item.lineno
                    m_end = getattr(item, "end_lineno", m_start)
                    args_str = ast.unparse(item.args)
                    elements.append(f"    def {item.name}({args_str}) (lines {m_start}-{m_end})")
            if not methods_found:
                elements.append("    (no methods defined)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f_start = node.lineno
            f_end = getattr(node, "end_lineno", f_start)
            args_str = ast.unparse(node.args)
            elements.append(f"  def {node.name}({args_str}) (lines {f_start}-{f_end})")

    if len(elements) == 1:
        return f"No functions or classes found in '{path_obj.name}'."

    return "\n".join(elements)


def find_imports(file_name: str) -> str:
    """List project and external imports used by a specific project file.
    Use this when asked what a file imports, e.g. 'What does main.py import?'.
    Args: file_name: The Python file to check, e.g. 'main.py'.
    Returns: A string listing project-file and external imports with imported names.
    """
    path_obj, error = _validate_py_file(file_name)
    if error or not path_obj:
        return error or "Unknown validation error."

    tree, parse_error = _parse_ast(path_obj)
    if parse_error or not tree:
        return parse_error or "Unable to parse file."

    project_stems = {p.stem for p in _get_project_py_files()}
    project_imports, external_imports = _get_file_imports(tree, project_stems)

    output = [f"Imports for {path_obj.name}:", "Project imports:"]
    if project_imports:
        for mod in sorted(project_imports.keys()):
            names = project_imports[mod]
            if names:
                output.append(f"  - {mod} ({', '.join(sorted(set(names)))})")
            else:
                output.append(f"  - {mod}")
    else:
        output.append("  (none)")

    output.append("External imports:")
    if external_imports:
        for mod in sorted(external_imports.keys()):
            names = external_imports[mod]
            if names:
                output.append(f"  - {mod} ({', '.join(sorted(set(names)))})")
            else:
                output.append(f"  - {mod}")
    else:
        output.append("  (none)")

    return "\n".join(output)


def who_imports(module_name: str) -> str:
    """Find which project files import a given module or file name.
    Use this when asked who imports a module, e.g. 'Which files import config.py?'.
    Args: module_name: Target module name, e.g. 'config.py' or 'config'.
    Returns: A string listing project files that import it, or a message if none.
    """
    if not module_name or not isinstance(module_name, str):
        return "Invalid module name: name cannot be empty."
    if ".." in module_name or "/" in module_name or "\\" in module_name:
        return f"Security error: Invalid module name '{module_name}'. Path separators and '..' are not allowed."

    target = module_name.strip()
    if target.endswith(".py"):
        target = target[:-3]

    py_files = _get_project_py_files()
    if not py_files:
        return f"No Python files found in project directory: {PROJECT_DIR}."

    importing_files: list[str] = []

    for file_path in py_files:
        tree, parse_error = _parse_ast(file_path)
        if parse_error or not tree:
            continue

        imported_from_target = False
        names_taken: list[str] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target or alias.name.startswith(f"{target}."):
                        imported_from_target = True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == target or node.module.startswith(f"{target}.")):
                    imported_from_target = True
                    names_taken.extend([alias.name for alias in node.names])

        if imported_from_target:
            if names_taken:
                names_str = f" (imports: {', '.join(sorted(set(names_taken)))})"
            else:
                names_str = ""
            importing_files.append(f"  - {file_path.name}{names_str}")

    if not importing_files:
        return f"No project files import '{module_name}'."

    return f"Project files that import '{target}':\n" + "\n".join(importing_files)


def import_graph_mermaid() -> str:
    """Generate a Mermaid diagram of import dependencies between project files.
    Use this when asked for an architecture diagram, e.g. 'Show the dependency graph'.
    Args: None.
    Returns: A string containing a Mermaid flowchart in a markdown code fence.
    """
    py_files = _get_project_py_files()
    if not py_files:
        return f"No Python files found in project directory: {PROJECT_DIR}."

    project_stems = {p.stem for p in py_files}
    edges: set[tuple[str, str]] = set()

    for file_path in py_files:
        src = file_path.stem
        tree, parse_error = _parse_ast(file_path)
        if parse_error or not tree:
            continue
        project_imports, _ = _get_file_imports(tree, project_stems)
        for target_mod in project_imports.keys():
            target_stem = target_mod.split(".")[0]
            if target_stem in project_stems and target_stem != src:
                edges.add((src, target_stem))

    lines = ["```mermaid", "flowchart TD"]
    lines += [f'    {p.stem}["{p.name}"]' for p in py_files]
    lines += [f"    {src} --> {dest}" for src, dest in sorted(edges)]
    lines.append("```")

    return "\n".join(lines)


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

