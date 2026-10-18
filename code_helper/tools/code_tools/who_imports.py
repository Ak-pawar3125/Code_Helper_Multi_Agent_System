"""who_imports.py — Find which project files import a given module or file name."""

from ._helpers import _get_project_py_files, _parse_ast

from ..common import PROJECT_DIR


def who_imports(module_name: str) -> str:
    """Find which project files import a given module or file name.
    Use this when asked who imports a module, e.g. 'Which files import config.py?'.
    Args: module_name: Target module name, e.g. 'config.py' or 'config'.
    Returns: A string listing project files that import it, or a message if none.
    """
    import ast

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
