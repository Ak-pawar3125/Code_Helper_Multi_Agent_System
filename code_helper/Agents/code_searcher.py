from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from ..tools.rag import rag_search

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
