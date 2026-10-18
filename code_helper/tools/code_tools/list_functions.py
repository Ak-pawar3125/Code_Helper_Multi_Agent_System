"""list_functions.py — List all classes, methods, and functions in a file."""

import ast

from ._helpers import _validate_py_file, _parse_ast


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
