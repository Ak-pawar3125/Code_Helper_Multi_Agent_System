from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .tools.rag import rag_search
from .tools.code_tools import (
    list_project_files,
    list_functions,
    find_imports,
    who_imports,
    import_graph_mermaid,
    get_project_overview,
    read_file,
)
from .tools.mcp_client import (
    mcp_read_file_or_fallback,
    mcp_mermaid_or_fallback,
    get_mcp_client,
)

model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")

# Code_Searcher Agent
code_searcher = Agent(
    model=model,
    name="code_searcher",
    description=(
        "You are the code search agent. Your job is to find WHERE something is located in code or implemented in code (e.g. 'where is x done?')."
    ),
    instruction=(
        "You MUST call rag_search exactly once with the user's question. "
        "If the result starts with 'Found', it contains real code from the project and you "
        "MUST answer from it. Result [1] is the best match. Name its file, its line numbers "
        "and its function, then explain in one or two sentences what it does. Mention a second "
        "location only if another result is clearly relevant. "
        "Scores are always low for this search, so never judge relevance by the score. "
        "Only if the result starts with NO_RESULTS or TOOL_ERROR, say so in one sentence. "
        "Never call the tool twice."
    ),
    tools=[rag_search],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

# Code_Explainer Agent
code_explainer = Agent(
    model=model,
    name="code_explainer",
    description=(
        "You are the code explainer agent. Your job is to read one whole file and explain it line by line or explain what it does (e.g. 'explain vector_store.py')."
    ),
    instruction=(
        "You are the code explainer worker. Follow these numbered steps strictly:\n"
        "1. CALL TOOL ONCE: Identify the requested file name (e.g. 'vector_store.py') and call 'mcp_read_file_or_fallback' exactly ONCE. Never call it more than once or for other files.\n"
        "2. STOP CALLING TOOLS: Once the tool returns the file content, do NOT make any further tool calls.\n"
        "3. EXPLAIN AND FINISH: Immediately write your explanation using ONLY the content returned by the tool:\n"
        "   - Explain the file's primary purpose.\n"
        "   - Explain key classes and main functions.\n"
        "   - Explain how it connects to other files in the project.\n"
        "   - If the file cannot be read or was not found, say exactly: 'I couldn't read the file.'\n"
        "4. TERMINATE: Output your explanation and stop immediately."
    ),
    tools=[mcp_read_file_or_fallback],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

# Architecture_mapper Agent
architecture_mapper = Agent(
    model=model,
    name="architecture_mapper",
    description=(
        "You are the architecture mapper agent. Your job is to explain codebase structure or answer structure-related questions: generate structure of project, which files exist, which functions inside a file, and which files import which other packages in the codebase (e.g. 'Which files import config.py?'). Also generates Mermaid dependency diagrams."
    ),
    instruction=(
      """Answer structure questions with exactly ONE tool call:
            - who imports X          -> who_imports(X)
            - what does a file import -> find_imports(file)
            - functions in a file    -> list_functions(file)
            - list files             -> list_project_files()
            - graph/diagram only if explicitly asked -> import_graph_mermaid()
            Always use the .py extension.
            After the tool returns, do not call another tool. Give a short answer:
            one sentence, then one bullet per item, copying names exactly. No extra
            paragraphs or diagrams unless a graph was requested (then use a ```mermaid
            block plus one sentence).
            A result with items is a success. Report an error only if the output starts
            with "TOOL_ERROR:". If nothing imports the module, say so in one sentence."""
    ),
    tools=[list_project_files, list_functions, find_imports, who_imports, import_graph_mermaid],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

# Onboarding_Guide Agent
onboarding_guide = Agent(
    model=model,
    name="onboarding_guide",
    description=(
        "You are the onboarding guide agent for the codebase. Your job is to provide an overview of the project, introduce the codebase to new users, explain where to start reading, and how to run it (e.g., 'I am new. Where do I start?', 'Give me a project overview', 'What does this project do?', 'How do I run this?')."
    ),
    instruction=(
       "You are an onboarding guide for a Python code project. "
        "Call get_project_overview exactly once. Read its result and produce "
        "a concise, well-formatted answer in Markdown. "
        "Use these headings: '## Project Overview', "
        "'## Recommended Reading Order', and '## How to Run'. "
        "Explain the project in 2-3 sentences. "
        "Present the reading order as a numbered list, with one short explanation "
        "for each file. Give the run command in a separate code block. "
        "Do not reproduce the raw tool output, JSON, escaped newline characters, "
        "the full main.py snippet, or repeated filenames. "
        "Do not call write_final_answer or any other tool. "
        "Use only information supported by get_project_overview. "
        "If the tool returns an error beginning with TOOL_ERROR:, "
        "explain that the overview could not be retrieved."
    ),
    tools=[get_project_overview],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

# Main Coordinator or Boss Agent
root_agent = Agent(
    model=model,
    name="coordinator",
    description=(
        "You are the Coordinator agent (Boss). Your job is to route each question to the right worker. Never answer questions yourself."
    ),
    instruction=(
        "You are the Coordinator (Root supervisor). Follow these numbered rules strictly in order:\n\n"
        "1. NEVER ANSWER DIRECTLY: You must never answer code, file, architecture, or project questions yourself. You have no tools.\n"
        "2. ROUTE TO EXACTLY ONE WORKER: Read the user's question and immediately transfer to the single most appropriate worker agent:\n"
        "   - Route to 'onboarding_guide' only if the user is new, asks where to start, asks for a project/codebase overview, asks what the project does, or asks how to run the project.\n"
        "   - Route to 'code_searcher' only if the user asks where something is in code, where a function/feature is implemented, or to find specific logic.\n"
        "   - Route to 'code_explainer' only if the user asks to explain a specific file (e.g., 'explain vector_store.py') or what a file does.\n"
        "   - Route to 'architecture_mapper' only if the user asks about list functions/ classes or of specific file or project, list project files, codebase structure, dependencies, which files import what, or asks for a diagram.\n"
        "3. SINGLE TRANSFER ONLY: Transfer to only one worker agent per user query. Do not transfer to multiple agents.\n"
        "4. DO NOT RETRANSFER: Once you initiate a transfer, do not intervene or re-transfer."
    ),
    sub_agents=[code_searcher, code_explainer, architecture_mapper, onboarding_guide],
    generate_content_config=types.GenerateContentConfig(temperature=0),
)

# Coordinator alias for backwards compatibility
coordinator = root_agent




