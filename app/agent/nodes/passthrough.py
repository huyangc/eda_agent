import asyncio
from typing import Optional

from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.logger import get_logger, req_id

logger = get_logger(__name__)


async def passthrough_node(state: AgentState) -> dict:
    """Handle non-EDA requests with a direct LLM call — no RAG."""
    rid = req_id(state.get("request_id", ""))
    logger.info("%s [passthrough      ]  RAG OFF  direct LLM call", rid)

    llm = get_llm(streaming=True)
    queue: Optional[asyncio.Queue] = state.get("stream_queue")  # type: ignore[assignment]

    full_content = ""
    async for chunk in llm.astream(list(state["messages"])):
        token: str = chunk.content or ""
        full_content += token
        if queue is not None:
            await queue.put(token)

    if queue is not None:
        await queue.put(None)  # sentinel

    logger.info("%s [passthrough      ]  chars=%d", rid, len(full_content))
    return {"final_response": full_content, "token_usage": {}}
