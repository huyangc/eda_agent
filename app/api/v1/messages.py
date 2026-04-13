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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.agent.graph import eda_graph
from app.agent.state import AgentState
from app.config import settings
from app.logger import get_logger, req_id
from app.streaming.sse import anthropic_stream
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
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content else ""


class AnthropicTool(BaseModel):
    name: str
    description: str = ""
    input_schema: dict = {}


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
    tools: list[AnthropicTool] = []

    def system_text(self) -> Optional[str]:
        return _extract_text(self.system) if self.system else None


# ---------------------------------------------------------------------------
# Message conversion: Anthropic → LangChain
# ---------------------------------------------------------------------------

def _to_lc_messages(req: MessagesRequest):
    """Convert Anthropic-format messages to LangChain BaseMessage list.

    Handles:
    - Plain string content
    - Content arrays with text / tool_use / tool_result blocks
    """
    result = []

    system_text = req.system_text()
    if system_text:
        result.append(SystemMessage(content=system_text))

    for m in req.messages:
        if m.role == "user":
            result.extend(_convert_user_message(m))
        elif m.role == "assistant":
            result.append(_convert_assistant_message(m))

    return result


def _convert_user_message(m: AnthropicMessage) -> list:
    """User messages may contain text OR tool_result blocks (or both)."""
    if isinstance(m.content, str):
        return [HumanMessage(content=m.content)]

    tool_results = [
        b for b in m.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    text_blocks = [
        b for b in m.content
        if isinstance(b, dict) and b.get("type") == "text"
    ]

    messages = []

    # tool_result blocks → LangChain ToolMessage
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, list):
            content = _extract_text(content)
        messages.append(ToolMessage(
            content=content,
            tool_call_id=tr.get("tool_use_id", ""),
        ))

    # leftover text (rare but possible)
    text = "".join(b.get("text", "") for b in text_blocks)
    if text:
        messages.append(HumanMessage(content=text))

    return messages or [HumanMessage(content=m.text())]


def _convert_assistant_message(m: AnthropicMessage) -> AIMessage:
    """Assistant messages may contain text and/or tool_use blocks."""
    if isinstance(m.content, str):
        return AIMessage(content=m.content)

    text = _extract_text(m.content)
    tool_use_blocks = [
        b for b in m.content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]

    if not tool_use_blocks:
        return AIMessage(content=text)

    # Build LangChain tool_calls
    tool_calls = [
        {
            "id": b.get("id", f"toolu_{i}"),
            "name": b["name"],
            "args": b.get("input", {}),
        }
        for i, b in enumerate(tool_use_blocks)
    ]
    return AIMessage(content=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------

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
        tools=[t.model_dump() for t in req.tools],
        intent=None,
        intent_confidence=0.0,
        detected_tool_namespace=None,
        raw_command_candidates=[],
        retrieved_docs=[],
        retrieval_summary=None,
        output_mode="unknown",
        final_response=None,
        response_content_blocks=None,
        token_usage={},
        stream_queue=None,
        trace_writer=trace_writer,
    )


# ---------------------------------------------------------------------------
# Non-streaming response builder
# ---------------------------------------------------------------------------

def _build_sync_response(final_state: dict, request_id: str, model: str) -> dict:
    """Build Anthropic-format response from final graph state."""
    content_blocks = final_state.get("response_content_blocks")

    if content_blocks is not None:
        # Tool-use response
        stop_reason = (
            "tool_use"
            if any(b.get("type") == "tool_use" for b in content_blocks)
            else "end_turn"
        )
    else:
        # Plain text response
        text = final_state.get("final_response") or ""
        content_blocks = [{"type": "text", "text": text}]
        stop_reason = "end_turn"

    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


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
    session_id = http_req.headers.get("x-session-id") or uuid.uuid4().hex[:16]
    request_id = f"msg_{uuid.uuid4().hex}"
    rid = req_id(request_id)
    t_start = time.monotonic()

    tw = TraceWriter(session_id=session_id, log_dir=settings.log_dir)
    tw.request(
        system=req.system_text(),
        messages=[{"role": m.role, "content": m.text()} for m in req.messages],
    )

    # Log any tool_result blocks sent back by the client this turn
    tool_results_this_turn = []
    for m in req.messages:
        if m.role == "user" and isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = _extract_text(content)
                    tool_results_this_turn.append({
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": content,
                    })
    if tool_results_this_turn:
        tw.tool_results(tool_results_this_turn)

    has_tools = bool(req.tools)
    mode_label = "stream" if req.stream else "sync"
    logger.info(
        "%s ── NEW REQUEST ── POST /v1/messages  [%s]  session=%s  tools=%d  user: %r",
        rid, mode_label, session_id, len(req.tools), _user_preview(req),
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
            try:
                await eda_graph.ainvoke(state)
                elapsed = time.monotonic() - t_start
                logger.info("%s ── DONE ── %.2fs  session=%s", rid, elapsed, session_id)
            except Exception as exc:
                elapsed = time.monotonic() - t_start
                logger.exception("%s ── ERROR ── %.2fs  session=%s  %s", rid, elapsed, session_id, exc)
                # Unblock the SSE consumer so the client doesn't hang forever
                await queue.put(None)

        asyncio.create_task(run_graph())

        return StreamingResponse(
            anthropic_stream(queue, request_id, req.model),
            media_type="text/event-stream",
            headers=sse_headers,
        )

    # Non-streaming
    final_state = await eda_graph.ainvoke(state)
    elapsed = time.monotonic() - t_start
    logger.info("%s ── DONE ── %.2fs  session=%s", rid, elapsed, session_id)

    return JSONResponse(
        content=_build_sync_response(final_state, request_id, req.model),
        headers={"X-Session-ID": session_id},
    )
