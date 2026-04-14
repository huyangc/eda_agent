# EDA Agent

An OpenAI-compatible backend for EDA (Electronic Design Automation) engineers. It wraps a LangGraph agent pipeline that transparently performs RAG retrieval over Primetime and Innovus command documentation, then generates QA answers or TCL code.

## Architecture

```
User (CLI)  →  FastAPI  →  LangGraph Agent
                               ├── intent_detector      # classify EDA vs general
                               ├── command_extractor    # extract namespace + commands
                               ├── rag_retriever        # HTTP call → external RAG service
                               ├── response_generator   # final LLM call (QA or TCL)
                               └── passthrough          # non-EDA direct answer

External RAG Service (not in this repo)
  POST /retrieve  {query, namespace, top_k}  →  {docs: [{content, metadata, score}]}
```

## Project Structure

```
eda_agent/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan hooks
│   ├── config.py                # All config via pydantic-settings (reads .env)
│   ├── logger.py                # Centralized logging with per-request tracing
│   ├── tracing.py               # Per-session JSONL trace writer
│   ├── api/
│   │   ├── router.py            # API router aggregation
│   │   ├── health.py            # GET /health
│   │   ├── log_viewer.py        # Log file viewer endpoint
│   │   └── v1/
│   │       ├── chat.py          # POST /v1/chat/completions (OpenAI format)
│   │       ├── messages.py      # POST /v1/messages (Anthropic format)
│   │       └── models.py        # GET /v1/models
│   ├── schemas/
│   │   └── openai_compat.py     # ChatCompletionRequest/Response/Chunk
│   ├── agent/
│   │   ├── state.py             # AgentState TypedDict — contract between all nodes
│   │   ├── graph.py             # LangGraph StateGraph (compiled singleton: eda_graph)
│   │   ├── routing.py           # Conditional edge functions
│   │   └── nodes/
│   │       ├── intent_detector.py     # Classify EDA vs general intent
│   │       ├── command_extractor.py   # Extract namespace + command candidates
│   │       ├── rag_retriever.py       # Call external RAG service
│   │       ├── response_generator.py  # Generate final QA/TCL response
│   │       └── passthrough.py         # Non-EDA requests + tool-use relay
│   ├── rag/
│   │   └── client.py            # Async HTTP client (singleton: RAGClient)
│   ├── llm/
│   │   └── factory.py           # LLM factory: OpenAI / Bedrock / Ollama
│   └── streaming/
│       └── sse.py               # Queue-based SSE generator (OpenAI + Anthropic formats)
├── rag_service/                 # Independent mock RAG service (v1 stub)
│   ├── main.py                  # FastAPI app, provides POST /retrieve + GET /health
│   ├── schemas.py               # RetrieveRequest / RetrieveResponse
│   ├── llm_retriever.py         # Calls OpenAI to generate plausible EDA doc chunks
│   └── config.py                # Independent configuration (port=9000)
├── tests/
│   └── test_api.py              # API integration tests
├── CLAUDE.md                    # Project instructions for Claude Code
├── SERVICES.md                  # Service documentation
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

## Key Features

### Dual API Compatibility

- **OpenAI Compatible** (`POST /v1/chat/completions`): Connect with any OpenAI-compatible client
- **Anthropic Compatible** (`POST /v1/messages`): Connect with Claude Code CLI via `ANTHROPIC_BASE_URL=http://localhost:8000 claude`

### Smart Intent Routing

The LangGraph agent automatically routes requests through different paths:

1. **Tool-bearing requests** (from Claude Code subagents): Direct passthrough without RAG
2. **EDA-related queries**: Full RAG pipeline with documentation retrieval
3. **General queries**: Direct LLM response without RAG overhead

### EDA Tool Support

- **Primetime** (Synopsys): Static Timing Analysis commands like `report_timing`, `report_constraint`, `get_cells`
- **Innovus** (Cadence): Place & Route commands like `place_design`, `route_design`, `floorPlan`

### Output Modes

- **QA mode**: Natural language answers with technical explanations
- **TCL mode**: Fenced TCL code blocks with inline comments

### Streaming Support

Queue-based SSE streaming for both OpenAI and Anthropic wire formats. Supports:
- Plain text token streaming
- Structured tool_use events (for Anthropic format)

## Configuration

All settings live in `app/config.py` (`Settings` class) and are read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` / `bedrock` / `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name for main responses |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_BASE_URL` | OpenAI default | Override for proxies |
| `BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-sonnet...` | Bedrock model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `RAG_SERVICE_URL` | `http://localhost:9000` | External RAG service base URL |
| `RAG_TOP_K` | `6` | Max docs retrieved per request |
| `RAG_SCORE_THRESHOLD` | `0.35` | Min similarity score to include a doc |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.7` | Below this → still tries EDA path |
| `MAX_COMMANDS_TO_RETRIEVE` | `5` | Max command candidates for RAG queries |
| `TOOL_USE_MODEL` | `qwen3.5-27b` | Fast model for tool-routing decisions |

## Setup

### Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env to fill in OPENAI_API_KEY and other settings

# Start the RAG service (mock v1)
python rag_service/main.py   # listens on :9000

# Start the main agent
python app/main.py           # listens on :8000
# or
uvicorn app.main:app --reload

# Alternatively, use the restart script to manage background processes:
./restart.sh
```

### Running Tests

```bash
pytest tests/ -v
```

## Usage

### Via Claude Code CLI

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

### Via OpenAI-Compatible Client

```bash
export OPENAI_BASE_URL=http://localhost:8000
# Use with any OpenAI SDK or compatible client
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/v1/messages` | POST | Anthropic-compatible messages API |
| `/v1/models` | GET | List available models |
| `/logs/{session_id}` | GET | View session traces |

### Request Extensions

Both chat and messages endpoints support EDA-specific extensions:

**Force EDA tool namespace:**
```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "How do I report timing?"}],
  "eda_tool": "primetime"  // "primetime" | "innovus" | "auto"
}
```

**Force output mode:**
```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Show me TCL for floorplan"}],
  "output_mode": "tcl"  // "qa" | "tcl" | "auto"
}
```

## RAG Service (Mock v1)

`rag_service/` is an independent FastAPI service implementing the EDA Agent's RAG interface contract. **Current implementation is a stub**: it uses OpenAI to generate plausible-looking EDA documentation snippets instead of querying a vector database.

### Mock Document Generation

The mock retriever (`rag_service/llm_retriever.py`) calls OpenAI to generate documentation chunks that look like they came from actual Primetime or Innovus reference manuals. Each chunk includes:

- `content`: Documentation text (2-6 sentences or code example)
- `metadata`: Tool name, command name, source type
- `score`: Relevance score (0.75-0.99)

### Real RAG Integration

To integrate a real vector database:

1. Implement your own RAG service at a custom URL
2. Update `RAG_SERVICE_URL` in `.env`
3. Ensure the service implements the contract:
   ```
   POST /retrieve
   Request:  { "query": str, "namespace": "primetime"|"innovus"|"both", "top_k": int }
   Response: { "docs": [{ "content": str, "metadata": dict, "score": float }] }
   ```

## Logging & Tracing

### Application Logs

Logs are written to both stdout and date-partitioned files:
```
logs/
└── 2026-04-13/
    └── eda_agent.log
```

Log format shows clear per-request traces:
```
[req-abc12] ── NEW REQUEST ── POST /v1/messages  user: "What does report_timing..."
[req-abc12] [intent_detector   ]  intent=eda       confidence=0.95  reason="mentions primetime command"
[req-abc12] [cmd_extractor     ]  namespace=primetime  commands=['report_timing']  mode=qa
[req-abc12] [rag_retriever     ]  RAG ON           namespace=primetime  queries=1  docs=3  top_score=0.94
[req-abc12] [response_gen      ]  mode=qa          streaming  chars=842
[req-abc12] ── DONE ── 4.21s
```

### Session Traces

Each request gets a JSONL trace file under `logs/YYYY-MM-DD/`:
```
logs/2026-04-13/
└── a1b2c3d4.jsonl  # {session_id}.jsonl
```

Each line is a self-contained JSON event:
```json
{"ts": "09:40:34.123", "session_id": "a1b2c3d4", "type": "request", "data": {...}}
{"ts": "09:40:53.210", "session_id": "a1b2c3d4", "type": "intent", "data": {...}}
{"ts": "09:41:15.441", "session_id": "a1b2c3d4", "type": "cmd_extract", "data": {...}}
{"ts": "09:41:15.890", "session_id": "a1b2c3d4", "type": "rag_query", "data": {...}}
{"ts": "09:41:16.320", "session_id": "a1b2c3d4", "type": "rag_response", "data": {...}}
{"ts": "09:42:07.001", "session_id": "a1b2c3d4", "type": "llm_response", "data": {...}}
```

## Design Decisions

1. **RAG is external**: `app/rag/client.py` is a thin HTTP client. The vector store, embeddings, and ingest pipeline are outside this repo.

2. **Per-command RAG queries**: `rag_retriever` issues one query per extracted command candidate, then merges and deduplicates results — better precision than a single combined query.

3. **Queue-based streaming**: `response_generator` pushes tokens into an `asyncio.Queue`; the SSE layer consumes it. This decouples the LangGraph graph from the HTTP transport.

4. **Two output modes**: `qa` (natural language) and `tcl` (fenced TCL code block). Detected automatically; overridable via request fields.

5. **Integrated Tool Use & RAG**: When Claude Code subagents call the backend, their tools are translated and bound to the backend LLM. If an EDA intent is detected, RAG context is injected *into* the tool-capable LLM prompt, empowering the LLM to generate `tool_use` instructions directly driven by RAG knowledge.

6. **Concurrent LLM throttling**: A global semaphore limits concurrent LLM API calls to avoid rate-limit timeouts when Claude Code fires many parallel tool-use requests.

7. **Client-Side Tool Execution (Red Line Architecture)**: The backend Agent *never* executes tools locally. It solely acts as the reasoning engine and RAG retriever—issuing `tool_use` JSON instructions via Server-Sent Events (SSE) back to the client. The client CLI (e.g. Claude Code) is entirely responsible for physically executing file/shell operations and returning the `tool_result` to the backend.

## License

Internal use only.
