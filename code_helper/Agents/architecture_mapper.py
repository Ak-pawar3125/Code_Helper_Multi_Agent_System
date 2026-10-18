from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from ..tools.code_tools import (
    list_project_files,
    list_functions,
    find_imports,
    who_imports,
    import_graph_mermaid,
)

model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")

# Architecture_mapper Agent
architecture_mapper = Agent(
    model=model,
    name="architecture_mapper",
    description=(
        "You are the architecture mapper agent. Your job is to explain codebase structure or answer structure-related questions and generates Mermaid dependency diagrams:"
        "generate structure of project, which files exist, which functions inside a file, and which files import which other packages in the codebase (e.g. 'Which files import config.py?')."
    ),
    instruction=(
     "You answer code structure questions. Call exactly ONE tool, exactly ONE time, then write "
    "your final answer in plain text from that tool's result. "
    "Pick the tool like this: "
    "'which files import X' -> who_imports('X.py'). "
    "'what does a file import' -> find_imports('file.py'). "
    "'what functions or classes are in a file' -> list_functions('file.py'). "
    "'which files are in the project' -> list_project_files(). "
    "'diagram', 'graph', 'flow' or 'how the files are connected' -> import_graph_mermaid(). "
    "Always use the .py extension in file names. "
    "After the tool returns, you already have everything you need. Never call a tool a second "
    "time, not even the same one. "
    "How to answer: one short sentence, then one bullet per item, copying names exactly as the "
    "result shows them. No extra paragraphs. "
    "For a diagram, put the ```mermaid block from the result in your answer, then one sentence. "
    "If the result is an error message, such as 'File ... not found' or one starting with "
    "TOOL_ERROR, repeat that message to the user and stop."
    ),
    tools=[list_project_files, list_functions, find_imports, who_imports, import_graph_mermaid],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)
