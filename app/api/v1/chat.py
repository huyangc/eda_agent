import asyncio
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.graph import eda_graph
from app.agent.state import AgentState
from app.schemas.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    UsageInfo,
)
from app.streaming.sse import stream_response

router = APIRouter()


def _to_lc_messages(messages: list[ChatMessage]):
    """Convert OpenAI-style messages to LangChain BaseMessage objects."""
    result = []
    for m in messages:
        if m.role == "system":
            result.append(SystemMessage(content=m.content))
        elif m.role == "user":
            result.append(HumanMessage(content=m.content))
        else:
            result.append(AIMessage(content=m.content))
    return result


def _build_initial_state(req: ChatCompletionRequest, request_id: str) -> AgentState:
    return AgentState(
        messages=_to_lc_messages(req.messages),
        request_id=request_id,
        forced_tool=req.eda_tool if req.eda_tool != "auto" else None,
        forced_mode=req.output_mode if req.output_mode != "auto" else None,
        tools=[],
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
        trace_writer=None,
    )


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    state = _build_initial_state(req, request_id)

    if req.stream:
        queue: asyncio.Queue = asyncio.Queue()
        state["stream_queue"] = queue

        async def run_graph() -> None:
            await eda_graph.ainvoke(state)

        asyncio.create_task(run_graph())

        return StreamingResponse(
            stream_response(queue, request_id, req.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    final_state = await eda_graph.ainvoke(state)
    content = final_state.get("final_response") or ""
    usage = final_state.get("token_usage") or {}

    response = ChatCompletionResponse(
        id=request_id,
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )
    return JSONResponse(content=response.model_dump())
