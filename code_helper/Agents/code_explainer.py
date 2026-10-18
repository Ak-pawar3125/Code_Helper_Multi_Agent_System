from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from ..tools.mcp_client import mcp_read_file_or_fallback

model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")

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
