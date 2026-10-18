"""find_imports.py — List project and external imports used by a specific project file."""

from ._helpers import _validate_py_file, _parse_ast, _get_project_py_files, _get_file_imports


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
