from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from ..tools.code_tools import get_project_overview

model = LiteLlm(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct")

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
