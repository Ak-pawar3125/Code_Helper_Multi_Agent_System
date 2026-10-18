# Code_Helper_Multi_Agent_System

### What Code_Helper does?

Code Helper answers questions about a small Python project (Sample_Project) using a team of five AI agents built with Google ADK: one coordinator(Boss) that only routes, and four workers(Code_Searcher, Code_Explainer, Architecture_Mapper, Onboarding_Guide). Each worker handles one kind of question: where code is, what a file does, how files connect, and where a newcomer should start. Search runs on a RAG pipeline written from scratch (no LangChain or LlamaIndex) that stores code chunks in Pinecone.

### Architecture of Code_Helper

![alt text](image.png)

### Agents

| Agent | Job | Tools | Example question |
|---|---|---|---|
| `coordinator` | Routes each question to the right worker. Never answers itself. | none | any question |
| `code_searcher` | Finds where something is implemented (RAG search). | `rag_search` | Where is the search done? |
| `code_explainer` | Reads a whole file and explains it simply. | Filesystem MCP (`mcp_read_file`) | Explain vector_store.py |
| `architecture_mapper` | Lists files, functions and imports, and draws the import map. | `list_project_files`, `list_functions`, `find_imports`, `who_imports`, `import_graph_mermaid` | Which files import config.py? |
| `onboarding_guide` | Explains what the project does, how to run it and where to start. | `get_project_overview` | I am new. Where do I start? |

### Tools

| Tool | Used by | Purpose | Uses AI? |
|---|---|---|---|
| `rag_search` | `code_searcher` | Returns the closest code chunks from Pinecone (file, lines, function name) | Embedding only |
| `list_project_files` | `architecture_mapper` | Lists `.py` files with line counts | No |
| `list_functions` | `architecture_mapper` | Lists classes, methods and functions in a file | No |
| `find_imports` | `architecture_mapper` | Shows what a file imports | No |
| `who_imports` | `architecture_mapper` | Shows which files import a module | No |
| `import_graph_mermaid` | `architecture_mapper` | Returns the import map as Mermaid text | No |
| `get_project_overview` | `onboarding_guide` | Summarizes each file and where to begin | No |
| Filesystem MCP tools | `code_explainer` | Read whole files (not counted in the 7 above) | No |

Six of the seven tools are plain Python.

### RAG pipeline

1. **Read:** all top-level `.py` files in `Sample_Project/`.
2. **Split:** structure-aware chunking (see below).
3. **Embed:** `nvidia/nemotron-3-embed-1b` on NVIDIA NIM via the `openai` client (2048 dimensions). Chunks are embedded as *passage*, questions as *query*.
4. **Store:** Pinecone serverless index `code-helper` (cosine), with file path, line range, chunk type, name and text as metadata.
5. **Search:** the question is embedded and Pinecone returns the 5 closest chunks.

### Chunking

**Method:** structure-aware chunking, built with Python's `ast` module and `tiktoken`. The code is split along its own structure (functions and classes) instead of at fixed sizes.

- **One chunk per function and per class.** Methods stay inside their class chunk.
- **Module-level code** (imports, constants, docstring) becomes its own chunk. Chunks under 60 tokens are skipped, since snippets like `if __name__ == "__main__": main()` matched unrelated questions.
- **300-token cap.** A block above the cap is split by lines with a 3-line overlap, so a cut never separates a statement from its context.

**Why:** a question like "Where is the search done?" is usually answered by one whole function. Keeping each function and class as one chunk makes every result readable on its own, and the saved line range and name point to an exact place. A fixed-size splitter would cut functions in half.

**Result on the sample project:** 49 chunks, averaging 205 tokens, with the largest at the 300-token cap.

### MCP server: Filesystem

**Server**: @modelcontextprotocol/server-filesystem, started with npx and connected through ADK's McpToolset. It is attached to code_explainer only.

**Why**: the Code Explainer must read whole files, and this ready-made server provides file-reading tools without writing a custom function. It can only read the folder it is given, so it is restricted to Sample_Project/ and cannot reach .env or the agent code.

### Sample Questions

#### Question 1: Search goal define in config.py file

Answer : The goal of the code is to load, validate, and manage credentials and runtime parameters for NVIDIA NIM (Embeddings & LLM Generation) and Pinecone Vector Database. This is done to centralize configuration, prevent scattered calls to os.getenv() across multiple modules, and guarantee that if an API key is missing or the network is unavailable during a live lecture, the system can gracefully inform the student or fall back to a safe offline demonstration mode without crashing.


#### Question 2: list functions of main.py.

Answer : The main.py file contains four functions:
print_banner(title: str)
print_scorecard(query: str, raw_answer: str, rag_answer: str, chunks: list[dict])
run_full_demo(config: RAGConfig)
main()

#### Question 3: Explain main.py file

Answer : The main.py file is the entry point of the Naive RAG Live Demonstration & Stage Walkthrough project. It provides an end-to-end interactive CLI runner for the Day 4 Naive RAG lecture, tracing each stage against the 8-box RAG architecture diagram (Slide 13).

The goal of the code is to load, validate, and manage credentials and runtime parameters for NVIDIA NIM (Embeddings & LLM Generation) and Pinecone Vector Database. This is done to centralize configuration, prevent scattered calls to os.getenv() across multiple modules, and guarantee that if an API key is missing or the network is unavailable during a live lecture, the system can gracefully inform the student or fall back to a safe offline demonstration mode without crashing.

The main.py file contains several functions, including:

print_banner: prints a high-visibility terminal header box for classroom projectors
print_scorecard: prints an educational scorecard comparing the raw model response to the RAG response
run_full_demo: executes the entire 8-stage live classroom walkthrough
main: parses command-line arguments and runs the chosen demonstration routine
The main function uses the argparse library to parse command-line arguments and run the chosen demonstration routine. It supports several options, including:

--index: runs only the offline indexing pipeline (Stages 1-4)
--query: runs only the online query pipeline with a custom question (Stages 5-8)
--compare: runs the baseline vs. RAG comparison on the canonical question
--mock: forces offline simulation mode without making external API calls
The main function also defines a canonical test query and uses it to run the complete live demo walkthrough by default.


### Troubleshooting: Agents Looping on Tool Calls

**Problem:** Worker agents called the same tool repeatedly (50-100 events for one question) and sometimes returned a false failure message such as "I couldn't map the architecture", even though the tool worked.

**Symptoms** (seen in the ADK Events trace):
- Tools like `who_imports`, `list_functions` and `get_project_overview` returned correct results, but the agent kept calling them again.
- Agents ended with failure messages when the output did not start with `TOOL_ERROR:`.
- Agents called tools the question did not need (e.g. drawing the import graph for a simple "who imports config.py").

**Cause:** The tools were fine. The model (Llama 3.2 11B) was unreliable after receiving a tool result: it did not treat the result as a final answer and fell back on failure wording. Workers that could transfer control to the coordinator or to each other likely added extra events.

**Fix:**
1. **Clear stop rule in instructions:** call the tool exactly once, then answer from the result. Treat only output starting with `TOOL_ERROR:` as a failure, and remove the canned failure sentences.
2. **`temperature=0`** for more consistent behaviour after tool calls.
3. **Transfers disabled on workers**, so they cannot hand control back or sideways and must answer themselves.