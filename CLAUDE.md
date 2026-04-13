# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDA Agent — an OpenAI-compatible backend for EDA (Electronic Design Automation) engineers. It wraps a LangGraph agent pipeline that transparently performs RAG retrieval over Primetime and Innovus command documentation, then generates QA answers or TCL code. Users connect via Claude Code CLI or any OpenAI-compatible client.

## Architecture

```
User (CLI)  →  FastAPI  →  LangGraph Agent
                               ├── intent_detector      classify EDA vs general
                               ├── command_extractor    extract namespace + commands
                               ├── rag_retriever        HTTP call → external RAG service
                               ├── response_generator   final LLM call (QA or TCL)
                               └── passthrough          non-EDA direct answer

External RAG Service (not in this repo)
  POST /retrieve  {query, namespace, top_k}  →  {docs: [{content, metadata, score}]}
```

## Project Structure

```
app/
├── main.py                  # FastAPI app factory + lifespan hooks
├── config.py                # All config via pydantic-settings (reads .env)
├── api/
│   ├── router.py
│   ├── health.py            # GET /health
│   └── v1/
│       ├── chat.py          # POST /v1/chat/completions (streaming + non-streaming)
│       └── models.py        # GET /v1/models
├── schemas/
│   └── openai_compat.py     # ChatCompletionRequest/Response/Chunk (mirrors OpenAI API)
├── agent/
│   ├── state.py             # AgentState TypedDict — contract between all nodes
│   ├── graph.py             # LangGraph StateGraph (compiled singleton: eda_graph)
│   ├── routing.py           # Conditional edge functions
│   └── nodes/
│       ├── intent_detector.py
│       ├── command_extractor.py
│       ├── rag_retriever.py
│       ├── response_generator.py
│       └── passthrough.py
├── rag/
│   └── client.py            # Async HTTP client (singleton: RAGClient)
├── llm/
│   └── factory.py           # LLM factory: OpenAI / Bedrock / Ollama
└── streaming/
    └── sse.py               # Queue-based SSE generator
tests/
└── test_api.py
```

## RAG Service (Mock v1)

`rag_service/` 是一个独立的 FastAPI 服务，实现 eda_agent 所需的 RAG 接口契约。**当前为伪实现**：不使用向量数据库，而是直接调用 OpenAI 生成看起来真实的 EDA 文档片段作为返回结果。

```
rag_service/
├── main.py          # FastAPI app，提供 POST /retrieve 和 GET /health
├── schemas.py       # RetrieveRequest / RetrieveResponse (与 eda_agent 契约一致)
├── llm_retriever.py # 调用 OpenAI 生成伪文档片段
└── config.py        # 独立配置 (port=9000, openai 设置)
```

**启动 RAG 服务：**
```bash
python rag_service/main.py   # 默认监听 :9000
```

**接口契约：**
```
POST /retrieve
请求: { "query": str, "namespace": "primetime"|"innovus"|"both", "top_k": int }
响应: { "docs": [{ "content": str, "metadata": { "tool", "command_name", ... }, "score": float }] }
```

## Development Setup

```bash
pip install -e ".[dev]"
cp .env.example .env         # fill in OPENAI_API_KEY and RAG_SERVICE_URL
python app/main.py           # starts on :8000
# or
uvicorn app.main:app --reload
```

## Configuration

All settings live in `app/config.py` (`Settings` class) and are read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` / `bedrock` / `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_BASE_URL` | OpenAI default | Override for proxies |
| `RAG_SERVICE_URL` | `http://localhost:9000` | External RAG service base URL |
| `RAG_TOP_K` | `6` | Max docs retrieved per request |
| `RAG_SCORE_THRESHOLD` | `0.35` | Min similarity score to include a doc |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.7` | Below this → still tries EDA path |
| `MAX_COMMANDS_TO_RETRIEVE` | `5` | Max command candidates for RAG queries |

## Running Tests

```bash
pytest tests/ -v
```

## Connecting via Claude Code CLI

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

Or set `OPENAI_BASE_URL=http://localhost:8000` with any OpenAI-compatible client.

## Key Design Decisions

- **RAG is external**: `app/rag/client.py` is a thin HTTP client. The vector store, embeddings, and ingest pipeline are outside this repo.
- **Per-command RAG queries**: `rag_retriever` issues one query per extracted command candidate, then merges and deduplicates results — better precision than a single combined query.
- **Queue-based streaming**: `response_generator` pushes tokens into an `asyncio.Queue`; the SSE layer consumes it. This decouples the LangGraph graph from the HTTP transport.
- **Two output modes**: `qa` (natural language) and `tcl` (fenced TCL code block). Detected automatically; overridable via the `output_mode` field in the request.
- **Supported EDA tools**: Primetime (STA) and Innovus (P&R). Namespace auto-detected; overridable via `eda_tool` field.
