from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .code_searcher import code_searcher
from .code_explainer import code_explainer
from .architecture_mapper import architecture_mapper
from .onboarding_guide import onboarding_guide

model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")

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
