# EDA Agent — 服务启动手册

本文档描述两个服务的完整启动流程，供手动操作或 Claude 托管时使用。

---

## 服务一览

| 服务 | 目录 | 默认端口 | 说明 |
|---|---|---|---|
| RAG Service | `rag_service/` | 9000 | 伪 RAG 接口，调用 LLM 生成 EDA 文档片段 |
| EDA Agent | `app/` | 8000 | OpenAI 兼容的 Agent 后端，含 LangGraph 流程 |

两个服务**相互独立**，EDA Agent 在处理 EDA 相关请求时会调用 RAG Service。**启动顺序：先启动 RAG Service，再启动 EDA Agent。**

---

## 环境准备（首次运行）

### 1. 安装依赖

```bash
cd /Users/hzf/workspace/eda_agent
pip install -e ".[dev]"
```

### 2. 配置环境变量

项目根目录下创建 `.env` 文件（两个服务共用同一份 `.env`）：

```bash
cp .env.example .env
```

当前配置（已在 `.env.example` 中设定）：

```ini
# LLM Backend（EDA Agent 使用）
LLM_BACKEND=openai
LLM_MODEL=deepseek-reasoner
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL=https://api.deepseek.com

# External RAG Service（EDA Agent 调用的地址）
RAG_SERVICE_URL=http://localhost:9000

# RAG Service 内部调用 LLM 也复用上面的 openai 配置
```

> RAG Service 与 EDA Agent 共用同一份 `.env`，`LLM_MODEL` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` 均复用，无需额外配置。

---

## 启动步骤

### 步骤 1 — 启动 RAG Service

```bash
cd /Users/hzf/workspace/eda_agent
python rag_service/main.py
```

**预期输出：**
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000
```

**健康检查：**
```bash
curl http://localhost:9000/health
# 预期: {"status":"ok"}
```

---

### 步骤 2 — 启动 EDA Agent

新开一个终端：

```bash
cd /Users/hzf/workspace/eda_agent
python app/main.py
```

**预期输出：**
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**健康检查：**
```bash
curl http://localhost:8000/health
# 预期: {"status":"ok"}
```

---

## 验证端到端流程

### 非流式请求

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eda-agent",
    "messages": [{"role": "user", "content": "What does report_timing do in PrimeTime?"}],
    "stream": false
  }' | python -m json.tool
```

### 流式请求

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eda-agent",
    "messages": [{"role": "user", "content": "写一个检查 setup violation 的 PrimeTime TCL 脚本"}],
    "stream": true
  }'
```

### 直接测试 RAG Service

```bash
curl -s -X POST http://localhost:9000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "report_timing setup slack",
    "namespace": "primetime",
    "top_k": 3
  }' | python -m json.tool
```

### 通过 Claude Code CLI 连接

```bash
OPENAI_BASE_URL=http://localhost:8000 OPENAI_API_KEY=any claude
```

---

## 停止服务

两个终端各自按 `Ctrl+C` 即可。

---

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| EDA Agent 返回的 retrieved_docs 为空 | RAG Service 未启动，或 `RAG_SERVICE_URL` 配置错误 | 确认 RAG Service 在 :9000 运行；检查 `.env` 中 `RAG_SERVICE_URL` |
| `401 Unauthorized` | API Key 无效 | 检查 `.env` 中 `OPENAI_API_KEY` |
| `Connection refused` on port 8000/9000 | 服务未启动 | 按顺序重新执行步骤 1、2 |
| RAG Service 返回空 docs | LLM 调用失败或返回格式异常 | 查看 RAG Service 终端日志；检查 `OPENAI_MODEL` 是否支持 `response_format: json_object` |
| `deepseek-reasoner` 响应慢 | 推理模型本身较慢 | 正常现象，可在 `.env` 改为 `deepseek-chat` 加速 |

---

## 文件结构速查

```
/Users/hzf/workspace/eda_agent/
├── .env                    ← 实际配置（不入库）
├── .env.example            ← 配置模板
├── app/main.py             ← EDA Agent 入口 (port 8000)
├── rag_service/main.py     ← RAG Service 入口 (port 9000)
└── SERVICES.md             ← 本文档
```
