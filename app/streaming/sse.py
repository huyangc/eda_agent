"""SSE streaming helpers for both OpenAI and Anthropic wire formats.

Queue protocol (shared by all nodes):
    str   — plain text token  (existing EDA / passthrough-text path)
    dict  — structured event  (tool_use path)
              {"type": "tool_start", "id": str, "name": str, "index": int}
              {"type": "tool_delta", "index": int, "partial_json": str}
              {"type": "tool_stop",  "index": int}
              {"type": "stop",       "stop_reason": "end_turn" | "tool_use"}
    None  — end-of-stream sentinel
"""

import asyncio
import json
import time
from typing import AsyncGenerator

from app.schemas.openai_compat import (
    ChatCompletionChunk,
    ChoiceDelta,
    StreamChoice,
)


# ---------------------------------------------------------------------------
# OpenAI SSE (used by /v1/chat/completions)
# ---------------------------------------------------------------------------

async def stream_response(
    queue: asyncio.Queue,
    request_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """Yield OpenAI-format SSE chunks from the token queue."""
    created = int(time.time())
    yield _make_openai_chunk(request_id, model, created, role="assistant", content="")

    while True:
        event = await queue.get()
        if event is None:
            yield _make_openai_chunk(request_id, model, created, content="", finish_reason="stop")
            yield "data: [DONE]\n\n"
            break
        if isinstance(event, str):
            yield _make_openai_chunk(request_id, model, created, content=event)
        # dict events are not used in the OpenAI path; silently ignore


def _make_openai_chunk(
    request_id: str,
    model: str,
    created: int,
    content: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
) -> str:
    chunk = ChatCompletionChunk(
        id=request_id,
        model=model,
        created=created,
        choices=[
            StreamChoice(
                index=0,
                delta=ChoiceDelta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


# ---------------------------------------------------------------------------
# Anthropic SSE (used by /v1/messages)
# ---------------------------------------------------------------------------

async def anthropic_stream(
    queue: asyncio.Queue,
    request_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """Yield Anthropic-format SSE events from the structured queue.

    Handles both plain text tokens (str) and tool_use events (dict).
    """
    created = int(time.time())

    yield _anth_event("message_start", {
        "type": "message_start",
        "message": {
            "id": request_id, "type": "message", "role": "assistant",
            "content": [], "model": model, "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _anth_event("ping", {"type": "ping"})

    # Track open content blocks: index → "text" | "tool_use"
    text_block_open = False
    text_block_index = 0
    stop_reason = "end_turn"

    while True:
        event = await queue.get()

        if event is None:
            break

        if isinstance(event, str):
            # ---- plain text token ----
            if not text_block_open:
                yield _anth_event("content_block_start", {
                    "type": "content_block_start", "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                text_block_open = True
            yield _anth_event("content_block_delta", {
                "type": "content_block_delta", "index": text_block_index,
                "delta": {"type": "text_delta", "text": event},
            })

        elif isinstance(event, dict):
            etype = event.get("type")

            if etype == "tool_start":
                # Close any open text block first
                if text_block_open:
                    yield _anth_event("content_block_stop", {
                        "type": "content_block_stop", "index": text_block_index,
                    })
                    text_block_open = False

                yield _anth_event("content_block_start", {
                    "type": "content_block_start",
                    "index": event["index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": event["id"],
                        "name": event["name"],
                        "input": {},
                    },
                })

            elif etype == "tool_delta":
                yield _anth_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": event["index"],
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": event["partial_json"],
                    },
                })

            elif etype == "tool_stop":
                yield _anth_event("content_block_stop", {
                    "type": "content_block_stop", "index": event["index"],
                })

            elif etype == "stop":
                stop_reason = event.get("stop_reason", "end_turn")

    # Close text block if still open
    if text_block_open:
        yield _anth_event("content_block_stop", {
            "type": "content_block_stop", "index": text_block_index,
        })

    yield _anth_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    yield _anth_event("message_stop", {"type": "message_stop"})


def _anth_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
