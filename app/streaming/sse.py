import asyncio
import time
from typing import AsyncGenerator

from app.schemas.openai_compat import (
    ChatCompletionChunk,
    ChoiceDelta,
    StreamChoice,
)


async def stream_response(
    queue: asyncio.Queue,
    request_id: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """Consume tokens from *queue* and yield SSE-formatted lines.

    The producer (response_generator / passthrough node) pushes str tokens and
    finally pushes ``None`` as a sentinel to signal end-of-stream.
    """
    created = int(time.time())

    # First chunk: role announcement
    yield _make_chunk(request_id, model, created, role="assistant", content="")

    while True:
        token = await queue.get()

        if token is None:
            # End-of-stream sentinel
            yield _make_chunk(request_id, model, created, content="", finish_reason="stop")
            yield "data: [DONE]\n\n"
            break

        yield _make_chunk(request_id, model, created, content=token)


def _make_chunk(
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
