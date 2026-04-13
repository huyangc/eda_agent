# EDA Agent 函数文档

## 项目概述

EDA Agent 是一个为电子设计自动化 (EDA) 工程师设计的 OpenAI/Anthropic 兼容后端服务。它使用 LangGraph 编排多节点智能体管道，对 Primetime 和 Innovus 命令文档执行透明的 RAG (检索增强生成)，然后生成 QA 答案或 TCL 代码。

---

## 目录结构

```
app/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置管理
├── logger.py            # 日志系统
├── tracing.py           # JSONL 追踪写入器
├── api/                 # API 路由层
│   ├── router.py        # 路由聚合
│   ├── health.py        # 健康检查
│   ├── log_viewer.py    # 日志查看器 UI
│   └── v1/
│       ├── chat.py      # OpenAI 格式聊天接口
│       ├── messages.py  # Anthropic 格式消息接口
│       └── models.py    # 模型列表接口
├── agent/               # LangGraph 智能体管道
│   ├── state.py         # 状态定义
│   ├── graph.py         # 图定义
│   ├── routing.py       # 条件路由
│   └── nodes/           # 处理节点
├── rag/
│   └── client.py        # RAG 服务 HTTP 客户端
├── llm/
│   └── factory.py       # LLM 工厂
├── streaming/
│   └── sse.py           # SSE 流式输出
└── schemas/
    └── openai_compat.py # OpenAI 兼容 Schema
```

---

## 核心模块函数详解

### app/main.py

| 函数 | 作用 |
|------|------|
| `create_app()` | 创建 FastAPI 应用实例，挂载路由和生命周期钩子 |
| `lifespan()` | 异步上下文管理器：启动时初始化 RAGClient，关闭时释放连接 |

---

### app/config.py

| 类/变量 | 作用 |
|---------|------|
| `LLMBackend` | LLM 后端枚举：OPENAI, BEDROCK, OLLAMA |
| `Settings` | Pydantic 设置类，从.env 读取配置 |
| `settings` | Settings 单例实例 |

**配置项：**
- `llm_backend`: LLM 后端选择
- `rag_service_url`: 外部 RAG 服务地址
- `rag_top_k`: RAG 检索最大文档数
- `intent_confidence_threshold`: 意图识别置信度阈值

---

### app/logger.py

| 函数 | 作用 |
|------|------|
| `setup_logging(level, log_dir)` | 初始化日志系统，输出到 stdout 和日期分片文件 |
| `get_logger(name)` | 获取命名日志记录器 |
| `req_id(request_id)` | 生成短请求 ID 前缀用于日志追踪 |

**日志格式示例：**
```
[req-abc123] ── NEW REQUEST ── POST /v1/messages
[req-abc123] [intent_detector ] eda (confidence=0.95)
[req-abc123] [response_gen    ] mode=qa streaming chars=842
[req-abc123] ── DONE ── 4.21s
```

---

### app/tracing.py

| 类 | 作用 |
|----|------|
| `TraceWriter` | 会话级追踪写入器，将结构化事件写入 JSONL 文件 |

**方法：**
- `__init__(session_id)`: 创建追踪写入器
- `log(event_type, data)`: 记录事件（线程安全）
- `close()`: 关闭文件句柄

**事件类型：** request, intent, cmd_extract, rag_query, rag_response, llm_response, tool_calls, tool_results

---

### app/api/router.py

| 变量 | 作用 |
|------|------|
| `api_router` | 主 API 路由器，聚合所有子路由 |

**路由注册：**
- `/health` → health_router
- `/logs/*` → log_viewer_router
- `/v1/chat/completions` → chat_router
- `/v1/messages/*` → messages_router
- `/v1/models` → models_router

---

### app/api/health.py

| 端点 | 作用 |
|------|------|
| `GET /health` | 返回 `{"status": "ok"}` 用于容器健康检查 |

---

### app/api/log_viewer.py

| 端点 | 作用 |
|------|------|
| `GET /logs` | 返回日志查看器 HTML 页面 |
| `GET /logs/api/dates` | 列出所有日期文件夹 |
| `GET /logs/api/sessions/{date}` | 列出指定日期的会话 |
| `GET /logs/api/events/{date}/{filename}` | 获取会话的完整事件列表 |

---

### app/api/v1/chat.py

| 函数 | 作用 |
|------|------|
| `create_chat_completion(req)` | OpenAI 格式聊天完成接口（支持流式/非流式） |
| `_to_lc_messages(messages)` | 将 OpenAI 消息转换为 LangChain BaseMessage |

**请求流程：**
1. 解析 ChatCompletionRequest
2. 初始化 AgentState
3. 提取 eda_tool/output_mode 覆盖参数
4. 流式模式：后台运行 eda_graph.ainvoke()，返回 SSE 流
5. 非流式模式：等待完成，返回完整响应

---

### app/api/v1/messages.py

| 函数 | 作用 |
|------|------|
| `create_message(req)` | Anthropic 格式消息接口（支持工具调用） |
| `stream_message(req)` | Anthropic 格式流式消息接口 |

**特性：**
- 完整支持 Anthropic tool_use 协议
- 支持 subagent 工具调用中继
- 事件流：message_start → content_block_* → message_delta → message_stop

---

### app/api/v1/models.py

| 端点 | 作用 |
|------|------|
| `GET /v1/models` | 返回可用模型列表 |

**虚拟模型：**
- `eda-agent`: 通用 EDA 助手
- `eda-agent-primetime`: Primetime 专用
- `eda-agent-innovus`: Innovus 专用

---

### app/schemas/openai_compat.py

| 类 | 作用 |
|----|------|
| `ChatMessage` | 聊天消息（role + content） |
| `ChatCompletionRequest` | 聊天完成请求（含 EDA 扩展字段） |
| `ChatCompletionResponse` | 聊天完成响应 |
| `ChatCompletionChunk` | 流式块 |
| `Choice`, `UsageInfo` | 响应子结构 |
| `ModelObject`, `ModelsListResponse` | 模型列表响应 |

**EDA 扩展字段：**
- `eda_tool`: "primetime" \| "innovus" \| "auto"
- `output_mode`: "qa" \| "tcl" \| "auto"

---

### app/agent/state.py

| 类 | 作用 |
|----|------|
| `AgentState` | TypedDict，定义节点间数据契约 |

**关键字段：**
- `messages`: 对话历史
- `request_id`: 请求 ID
- `intent`, `intent_confidence`: 意图识别结果
- `detected_tool_namespace`: 检测到的工具命名空间
- `raw_command_candidates`: 命令候选列表
- `retrieved_docs`, `retrieval_summary`: RAG 结果
- `output_mode`: 输出模式 (qa/tcl)
- `final_response`: 最终响应
- `stream_queue`: 流式队列
- `trace_writer`: 追踪写入器

---

### app/agent/graph.py

| 变量/函数 | 作用 |
|-----------|------|
| `eda_graph` | 编译后的 StateGraph 单例 |
| `build_graph()` | 构建并返回 StateGraph |

**图结构：**
```
Entry → [has tools?] → passthrough (短路)
        ↓
    intent_detector → [general?] → passthrough
                      ↓
                  command_extractor → [namespace=none?] → passthrough
                                      ↓
                                  rag_retriever → response_generator → END
```

---

### app/agent/routing.py

| 函数 | 作用 |
|------|------|
| `route_after_intent(state)` | 根据意图决定下一节点 |
| `route_after_command_extractor(state)` | 根据命名空间决定下一节点 |

**路由逻辑：**
- intent=general → passthrough
- intent=eda/ambiguous → command_extractor
- namespace=none → passthrough
- namespace=primetime/innovus/both → rag_retriever

---

### app/agent/nodes/intent_detector.py

| 函数 | 作用 |
|------|------|
| `intent_detector_node(state)` | 分类用户请求是否为 EDA 相关 |

**输出：**
- `intent`: "eda" \| "general" \| "ambiguous"
- `intent_confidence`: 0.0-1.0 置信度

---

### app/agent/nodes/command_extractor.py

| 函数 | 作用 |
|------|------|
| `command_extractor_node(state)` | 提取 EDA 工具命名空间和命令候选 |

**输出：**
- `detected_tool_namespace`: "primetime" \| "innovus" \| "both" \| "none"
- `raw_command_candidates`: 命令关键词列表
- `output_mode`: "qa" \| "tcl"

---

### app/agent/nodes/rag_retriever.py

| 函数 | 作用 |
|------|------|
| `rag_retriever_node(state)` | 调用外部 RAG 服务检索文档 |

**流程：**
1. 对每个命令候选构造查询："{candidate} {user_query}"
2. 调用 RAGClient.retrieve()
3. 按内容指纹去重
4. 按分数排序，取 top K，过滤低于阈值的
5. 生成 retrieval_summary 文本

---

### app/agent/nodes/response_generator.py

| 函数 | 作用 |
|------|------|
| `response_generator_node(state)` | 使用检索上下文生成最终响应 |

**模式：**
- **QA 模式**: 自然语言回答
- **TCL 模式**: 仅输出 fenced TCL 代码块

**流式支持：** 通过 stream_queue 推送 token

---

### app/agent/nodes/passthrough.py

| 函数 | 作用 |
|------|------|
| `passthrough_node(state)` | 处理非 EDA 请求或工具调用中继 |

**两种模式：**
1. **纯文本模式** (无 tools): 简单流式 LLM 调用
2. **工具中继模式** (有 tools): 
   - 转换 Anthropic 工具为 OpenAI 格式
   - 中继 tool_use 块回客户端
   - 压缩超长 tool_result
   - 滑动窗口消息修剪

---

### app/rag/client.py

| 类 | 作用 |
|----|------|
| `RAGClient` | RAG 服务 HTTP 客户端（单例） |

**方法：**
- `get_instance()`: 获取单例实例
- `retrieve(query, namespace, top_k)`: 异步检索文档
- `aclose()`: 关闭 HTTP 客户端

**API 契约：**
```
POST /retrieve
Request:  { "query": str, "namespace": str, "top_k": int }
Response: { "docs": [{"content": str, "metadata": dict, "score": float}] }
```

---

### app/llm/factory.py

| 函数 | 作用 |
|------|------|
| `get_llm(streaming=False, json_mode=False)` | 根据配置返回 LangChain 聊天模型 |

**支持后端：**
- **OpenAI**: ChatOpenAI
- **Bedrock**: ChatBedrock
- **Ollama**: ChatOllama

---

### app/streaming/sse.py

| 函数 | 作用 |
|------|------|
| `stream_response(queue, request_id, model)` | OpenAI SSE 格式流 |
| `anthropic_stream(queue, request_id)` | Anthropic SSE 格式流 |

**队列协议：**
- `str`: 纯文本 token
- `dict`: 结构化事件 (tool_start, tool_delta, tool_stop, stop)
- `None`: 流结束标记

---

## rag_service/ (Mock RAG 服务)

独立 FastAPI 服务，端口 9000，当前为伪实现。

| 文件 | 作用 |
|------|------|
| `main.py` | FastAPI 应用，提供 /retrieve 和 /health |
| `schemas.py` | RetrieveRequest / RetrieveResponse |
| `llm_retriever.py` | 调用 OpenAI 生成伪文档片段 |
| `config.py` | 独立配置 |

---

## 启动方式

```bash
# 启动 RAG 服务 (端口 9000)
python rag_service/main.py

# 启动 EDA Agent (端口 8000)
python app/main.py
# 或
uvicorn app.main:app --reload
```

## 客户端连接

```bash
# Claude Code CLI
ANTHROPIC_BASE_URL=http://localhost:8000 claude

# OpenAI 兼容客户端
OPENAI_BASE_URL=http://localhost:8000 ...
```
