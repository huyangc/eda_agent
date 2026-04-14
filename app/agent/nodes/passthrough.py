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

_MAX_TOOL_RESULT_CHARS = 80_000   # truncate total payload sent to LLM (DeepSeek-friendly)
_MAX_TOTAL_MSG_CHARS  = 80_000   # hard cap on total message history sent to LLM
_MAX_CONCURRENT_LLM   = 10       # max concurrent LLM API calls (DeepSeek rate-limit friendly)

logger = get_logger(__name__)

# Global semaphore to throttle concurrent DeepSeek API calls.
# Claude Code CLI fires many parallel requests — without throttling
# they queue up on DeepSeek's side and time out.
_llm_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """"Lazily create the semaphore in the running event loop."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)
    return _llm_semaphore


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
# Helpers
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    """Return True for errors that are worth a single retry."""
    name = type(exc).__name__
    return "Connection" in name or "Timeout" in name or "RateLimit" in name


async def _compress_tool_results(messages: list, rid: str) -> list:
    """Compress ToolMessage content if the total payload is too large.

    Instead of hard truncation, use a fast LLM to summarize the oversized output,
    preserving key information for the main reasoning model.
    """
    from langchain_core.messages import ToolMessage as LCToolMessage
    from langchain_core.messages import HumanMessage
    from app.config import settings

    total = sum(len(m.content) for m in messages if isinstance(m, LCToolMessage))
    if total <= _MAX_TOOL_RESULT_CHARS:
        return messages

    logger.warning(
        "%s [passthrough      ]  tool_result payload too large (%d chars) — compressing",
        rid, total,
    )

    # Budget: distribute _MAX_TOOL_RESULT_CHARS equally across all ToolMessages
    tool_msgs = [m for m in messages if isinstance(m, LCToolMessage)]
    per_msg = _MAX_TOOL_RESULT_CHARS // max(len(tool_msgs), 1)

    fast_llm = get_llm(
        model=settings.tool_use_model, 
        temperature=0.0, 
        timeout=20.0,
        max_retries=1
    )
    compressed = []
    
    for m in messages:
        if isinstance(m, LCToolMessage) and len(m.content) > per_msg:
            prompt = (
                f"You are an expert system summarizer. The following is output from a tool/command. "
                f"It is too large and needs to be compressed. "
                f"Extract the most essential information (IDs, paths, errors, metrics, key values). "
                f"Discard boilerplate, repetitive logs, formatting, and noise. KEEP IT CONCISE.\n\n"
                f"<tool_output>\n{m.content[:per_msg * 4]}\n</tool_output>\n\n"
                f"Output ONLY the compressed summary."
            )
            try:
                async with _get_semaphore():
                    summary = await fast_llm.ainvoke([HumanMessage(content=prompt)])
                new_content = str(summary.content) + f"\n... [LLM Compressed, original length {len(m.content)}]"
                compressed.append(LCToolMessage(content=new_content, tool_call_id=m.tool_call_id))
            except Exception as e:
                logger.warning("%s [passthrough      ]  compression failed, truncating instead: %s", rid, e)
                new_content = m.content[:per_msg] + f"\n... [truncated, original length {len(m.content)}]"
                compressed.append(LCToolMessage(content=new_content, tool_call_id=m.tool_call_id))
        else:
            compressed.append(m)
            
    return compressed


def _trim_message_history(messages: list, rid: str) -> list:
    """Sliding-window trim: keep system + the most recent messages within budget.

    Prevents total message payload from growing unboundedly across tool-use
    turns.  Always preserves the first message (system prompt) and the last
    few messages so the LLM has enough context.
    """
    total = sum(len(getattr(m, 'content', '') or '') for m in messages)
    if total <= _MAX_TOTAL_MSG_CHARS:
        return messages

    logger.warning(
        "%s [passthrough      ]  total message history too large (%d chars) — trimming",
        rid, total,
    )

    # Strategy: keep the first message (system) and drop the oldest
    # middle messages until we fit.
    if len(messages) <= 2:
        return messages

    head = messages[:1]        # system prompt
    tail = list(messages[1:])  # everything else

    while tail and sum(len(getattr(m, 'content', '') or '') for m in head + tail) > _MAX_TOTAL_MSG_CHARS:
        tail.pop(0)  # drop the oldest non-system message
        # Keep dropping until we hit a HumanMessage, to avoid breaking the 
        # AIMessage(tool_calls) -> ToolMessage chain which causes API 400 errors.
        while tail and getattr(tail[0], "type", "") != "human":
            tail.pop(0)

    if not tail:
        # Extreme case: even a single message exceeds the budget — keep last one

        tail = [messages[-1]]

    return head + tail


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

    async with _get_semaphore():
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
    custom_messages: list | None = None,
) -> dict:
    """Tool-use relay: bind tools, call LLM, push tool_use events into queue."""
    logger.info(
        "%s [passthrough      ]  RAG OFF  mode=tool_use  tools=%s",
        rid, [t["name"] for t in tools],
    )

    from app.config import settings
    oai_tools = _to_openai_tools(tools)
    llm = get_llm(
        streaming=False, 
        model=settings.tool_use_model, 
        timeout=180.0, 
        max_retries=0
    ).bind_tools(oai_tools)  # type: ignore[attr-defined]

    messages = custom_messages if custom_messages is not None else list(state["messages"])
    # Bypass all trimming/truncation per user request (retaining all history for huge context models like Qwen)
    # messages = _trim_message_history(messages, rid)

    # Check payload size before calling LLM — fail fast if too large even after trimming
    total_chars = sum(len(getattr(m, 'content', '') or '') for m in messages)

    # Retry once on transient connection errors
    msg: AIMessage
    for attempt in range(2):
        try:
            async with _get_semaphore():
                msg = await llm.ainvoke(messages)
            break
        except Exception as exc:
            if attempt == 0 and _is_transient(exc):
                # Don't retry if the payload is too large — it will just time out again
                if total_chars > _MAX_TOTAL_MSG_CHARS:
                    logger.warning(
                        "%s [passthrough      ]  payload too large (%d chars) for retry — failing fast",
                        rid, total_chars,
                    )
                else:
                    logger.warning("%s [passthrough      ]  connection error (attempt 1), retrying: %s", rid, exc)
                    await asyncio.sleep(2)
                    continue
            # Final failure — return an error text so the client doesn't see an empty end_turn
            err_text = f"[Backend error: {type(exc).__name__}. Please try again.]"
            logger.error("%s [passthrough      ]  LLM call failed: %s", rid, exc)
            if queue is not None:
                await queue.put(err_text)
                await queue.put({"type": "stop", "stop_reason": "end_turn"})
                await queue.put(None)
            return {
                "final_response": err_text,
                "response_content_blocks": [{"type": "text", "text": err_text}],
                "token_usage": {},
            }

    content_blocks = _build_content_blocks(msg)
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content_blocks) else "end_turn"
    full_text = "".join(b.get("text", "") for b in content_blocks if b["type"] == "text")

    # Guard: empty response (no text, no tool_calls) causes Claude Code to hang
    # because the Anthropic protocol requires at least one content block.
    if not content_blocks:
        logger.warning("%s [passthrough      ]  EMPTY response from LLM — injecting fallback text", rid)
        fallback = "I've processed the information. How can I help you further?"
        content_blocks = [{"type": "text", "text": fallback}]
        full_text = fallback
        msg.content = fallback

    logger.info(
        "%s [passthrough      ]  stop_reason=%s  blocks=%d",
        rid, stop_reason, len(content_blocks),
    )

    if queue is not None:
        await _push_ai_message(msg, queue)
        await queue.put(None)

    if tw := state.get("trace_writer"):
        tool_calls_made = getattr(msg, "tool_calls", []) or []
        if tool_calls_made:
            tw.tool_calls([
                {"id": tc.get("id"), "name": tc["name"], "args": tc.get("args", {})}
                for tc in tool_calls_made
            ])
        if full_text:
            tw.llm_response("passthrough_tool_use", full_text)

    return {
        "final_response": full_text,
        "response_content_blocks": content_blocks,
        "token_usage": {},
    }
