"""import_graph_mermaid.py — Generate a Mermaid diagram of import dependencies between project files."""

from ._helpers import _get_project_py_files, _parse_ast, _get_file_imports

from ..common import PROJECT_DIR


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
