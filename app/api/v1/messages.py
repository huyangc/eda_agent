"""Anthropic Messages API compatible endpoint.

Allows Claude Code CLI to connect via:
    ANTHROPIC_BASE_URL=http://localhost:8000 claude
"""

import asyncio
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.graph import eda_graph
from app.agent.state import AgentState
from app.config import settings
from app.logger import get_logger, req_id
from app.streaming.sse import stream_response
from app.tracing import TraceWriter

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request schema (Anthropic format)
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Accept content as plain string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]

    def text(self) -> str:
        return _extract_text(self.content)


class MessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    stream: bool = False
    system: Optional[str | list[dict]] = None
    temperature: Optional[float] = None

    def system_text(self) -> Optional[str]:
        return _extract_text(self.system) if self.system else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_lc_messages(req: MessagesRequest):
    result = []
    system_text = req.system_text()
    if system_text:
        result.append(SystemMessage(content=system_text))
    for m in req.messages:
        text = m.text()
        if m.role == "user":
            result.append(HumanMessage(content=text))
        else:
            result.append(AIMessage(content=text))
    return result


def _build_initial_state(
    req: MessagesRequest,
    request_id: str,
    trace_writer: TraceWriter,
) -> AgentState:
    return AgentState(
        messages=_to_lc_messages(req),
        request_id=request_id,
        forced_tool=None,
        forced_mode=None,
        intent=None,
        intent_confidence=0.0,
        detected_tool_namespace=None,
        raw_command_candidates=[],
        retrieved_docs=[],
        retrieval_summary=None,
        output_mode="unknown",
        final_response=None,
        token_usage={},
        stream_queue=None,
        trace_writer=trace_writer,
    )


# ---------------------------------------------------------------------------
# Streaming: Anthropic SSE event format
# ---------------------------------------------------------------------------

async def _anthropic_stream(
    queue: asyncio.Queue,
    request_id: str,
    model: str,
):
    """Convert token queue into Anthropic-format SSE events."""
    created = int(time.time())

    # message_start
    yield (
        f"event: message_start\n"
        f"data: {{"
        f'"type":"message_start","message":{{'
        f'"id":"{request_id}","type":"message","role":"assistant",'
        f'"content":[],"model":"{model}","stop_reason":null,'
        f'"stop_sequence":null,"usage":{{"input_tokens":0,"output_tokens":0}}'
        f"}}}}\n\n"
    )

    # content_block_start
    yield (
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
    )

    # ping
    yield 'event: ping\ndata: {"type":"ping"}\n\n'

    # token deltas
    while True:
        token = await queue.get()
        if token is None:
            break
        escaped = token.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        yield (
            f'event: content_block_delta\n'
            f'data: {{"type":"content_block_delta","index":0,'
            f'"delta":{{"type":"text_delta","text":"{escaped}"}}}}\n\n'
        )

    # content_block_stop
    yield 'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'

    # message_delta
    yield (
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
        '"stop_sequence":null},"usage":{"output_tokens":0}}\n\n'
    )

    # message_stop
    yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def _user_preview(req: MessagesRequest, max_len: int = 60) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            text = m.text().replace("\n", " ")
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return ""


@router.post("/messages")
async def create_message(req: MessagesRequest, http_req: Request):
    # Session ID: use X-Session-ID header if provided, otherwise generate one
    session_id = http_req.headers.get("x-session-id") or uuid.uuid4().hex[:16]
    request_id = f"msg_{uuid.uuid4().hex}"
    rid = req_id(request_id)
    t_start = time.monotonic()

    # Build trace writer and record the full incoming request
    tw = TraceWriter(session_id=session_id, log_dir=settings.log_dir)
    tw.request(
        system=req.system_text(),
        messages=[{"role": m.role, "content": m.text()} for m in req.messages],
    )

    mode_label = "stream" if req.stream else "sync"
    logger.info(
        "%s ── NEW REQUEST ── POST /v1/messages  [%s]  session=%s  user: %r",
        rid, mode_label, session_id, _user_preview(req),
    )

    state = _build_initial_state(req, request_id, tw)
    sse_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Session-ID": session_id,
    }

    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        state["stream_queue"] = queue

        async def run_graph() -> None:
            await eda_graph.ainvoke(state)
            elapsed = time.monotonic() - t_start
            logger.info("%s ── DONE ── %.2fs  session=%s", rid, elapsed, session_id)

        asyncio.create_task(run_graph())

        return StreamingResponse(
            _anthropic_stream(queue, request_id, req.model),
            media_type="text/event-stream",
            headers=sse_headers,
        )

    # Non-streaming
    final_state = await eda_graph.ainvoke(state)
    content = final_state.get("final_response") or ""
    elapsed = time.monotonic() - t_start
    logger.info("%s ── DONE ── %.2fs  session=%s", rid, elapsed, session_id)

    return JSONResponse(
        content={
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
            "model": req.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        headers={"X-Session-ID": session_id},
    )
