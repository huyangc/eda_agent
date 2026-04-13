"""Passthrough node — handles non-EDA requests.

Two sub-paths:
  • No tools  → plain text streaming (existing behaviour)
  • With tools → bind tools to LLM, relay tool_use blocks back to client
"""

import asyncio
import json
from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.logger import get_logger, req_id

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool format conversion
# ---------------------------------------------------------------------------

def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in anthropic_tools
    ]


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

async def _push_ai_message(msg: AIMessage, queue: asyncio.Queue, tool_offset: int = 1) -> str:
    """Push an AIMessage (text + optional tool_calls) into the queue.

    Returns the effective stop_reason: 'tool_use' if tool calls present, else 'end_turn'.
    """
    # Text content
    text = msg.content if isinstance(msg.content, str) else ""
    if isinstance(msg.content, list):
        text = "".join(
            b.get("text", "") for b in msg.content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if text:
        await queue.put(text)

    # Tool calls
    tool_calls = getattr(msg, "tool_calls", []) or []
    for i, tc in enumerate(tool_calls):
        index = tool_offset + i
        await queue.put({
            "type": "tool_start",
            "id": tc.get("id", f"toolu_{i}"),
            "name": tc["name"],
            "index": index,
        })
        partial = json.dumps(tc.get("args", {}), ensure_ascii=False)
        await queue.put({"type": "tool_delta", "index": index, "partial_json": partial})
        await queue.put({"type": "tool_stop", "index": index})

    stop_reason = "tool_use" if tool_calls else "end_turn"
    await queue.put({"type": "stop", "stop_reason": stop_reason})
    return stop_reason


def _build_content_blocks(msg: AIMessage) -> list[dict]:
    """Build Anthropic-format content blocks from an AIMessage (non-streaming)."""
    blocks: list[dict] = []

    text = msg.content if isinstance(msg.content, str) else ""
    if isinstance(msg.content, list):
        text = "".join(
            b.get("text", "") for b in msg.content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if text:
        blocks.append({"type": "text", "text": text})

    for i, tc in enumerate(getattr(msg, "tool_calls", []) or []):
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{i}"),
            "name": tc["name"],
            "input": tc.get("args", {}),
        })

    return blocks


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def passthrough_node(state: AgentState) -> dict:
    """Handle non-EDA requests — with or without tool use."""
    rid = req_id(state.get("request_id", ""))
    tools: list[dict] = state.get("tools") or []
    queue: Optional[asyncio.Queue] = state.get("stream_queue")  # type: ignore

    if tools:
        return await _passthrough_with_tools(state, tools, queue, rid)
    else:
        return await _passthrough_text(state, queue, rid)


async def _passthrough_text(state: AgentState, queue, rid: str) -> dict:
    """Plain text streaming — no tools."""
    logger.info("%s [passthrough      ]  RAG OFF  mode=text", rid)

    llm = get_llm(streaming=True)
    full_content = ""

    async for chunk in llm.astream(list(state["messages"])):
        token: str = chunk.content or ""
        full_content += token
        if queue is not None:
            await queue.put(token)

    if queue is not None:
        await queue.put(None)

    logger.info("%s [passthrough      ]  chars=%d", rid, len(full_content))
    if tw := state.get("trace_writer"):
        tw.llm_response("passthrough", full_content)

    return {
        "final_response": full_content,
        "response_content_blocks": None,
        "token_usage": {},
    }


async def _passthrough_with_tools(
    state: AgentState,
    tools: list[dict],
    queue,
    rid: str,
) -> dict:
    """Tool-use relay: bind tools, call LLM, push tool_use events into queue."""
    logger.info(
        "%s [passthrough      ]  RAG OFF  mode=tool_use  tools=%s",
        rid, [t["name"] for t in tools],
    )

    oai_tools = _to_openai_tools(tools)
    llm = get_llm(streaming=False).bind_tools(oai_tools)  # type: ignore[attr-defined]

    msg: AIMessage = await llm.ainvoke(list(state["messages"]))

    content_blocks = _build_content_blocks(msg)
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content_blocks) else "end_turn"
    full_text = "".join(b.get("text", "") for b in content_blocks if b["type"] == "text")

    logger.info(
        "%s [passthrough      ]  stop_reason=%s  blocks=%d",
        rid, stop_reason, len(content_blocks),
    )

    if queue is not None:
        await _push_ai_message(msg, queue)
        await queue.put(None)

    if tw := state.get("trace_writer"):
        tw.llm_response("passthrough_tool_use", full_text)

    return {
        "final_response": full_text,
        "response_content_blocks": content_blocks,
        "token_usage": {},
    }
