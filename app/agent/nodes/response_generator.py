import asyncio
from typing import Optional

from langchain_core.messages import SystemMessage

from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.logger import get_logger, req_id

logger = get_logger(__name__)

_QA_SYSTEM = """You are an expert EDA (Electronic Design Automation) assistant \
specializing in {namespace} tools.

Use the retrieved documentation below to answer the user's question accurately. \
If the documentation does not cover the question, answer from your own knowledge \
but clearly note the limitation.

--- Retrieved Documentation ---
{retrieval_context}
--- End Documentation ---

Answer in clear technical prose. Reference specific command options and flags when relevant."""

_TCL_SYSTEM = """You are an expert EDA TCL script writer for {namespace}.

Use the retrieved documentation below to generate correct, runnable TCL code.

--- Retrieved Documentation ---
{retrieval_context}
--- End Documentation ---

Output ONLY valid TCL code inside a fenced code block (```tcl ... ```). \
Add inline comments to explain non-obvious steps. \
Do not include explanatory prose outside the code block."""


async def response_generator_node(state: AgentState) -> dict:
    """Generate the final response using the LLM, optionally streaming."""
    namespace = state.get("detected_tool_namespace") or "EDA"
    output_mode = state.get("output_mode", "qa")
    retrieval_context = state.get("retrieval_summary") or "(No documentation retrieved.)"

    template = _TCL_SYSTEM if output_mode == "tcl" else _QA_SYSTEM
    system_prompt = template.format(
        namespace=namespace.upper() if namespace != "both" else "Primetime and Innovus",
        retrieval_context=retrieval_context,
    )

    # Build message list: system + conversation history
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    llm = get_llm(streaming=True)
    queue: Optional[asyncio.Queue] = state.get("stream_queue")  # type: ignore[assignment]

    full_content = ""
    usage: dict = {}

    async for chunk in llm.astream(messages):
        token: str = chunk.content or ""
        full_content += token
        if queue is not None:
            await queue.put(token)

    if queue is not None:
        await queue.put(None)  # sentinel: stream complete

    rid = req_id(state.get("request_id", ""))
    stream_label = "streaming" if queue is not None else "sync"
    logger.info(
        "%s [response_gen     ]  mode=%-5s  %s  chars=%d",
        rid, output_mode, stream_label, len(full_content),
    )
    if tw := state.get("trace_writer"):
        tw.llm_response(output_mode, full_content)
    return {
        "final_response": full_content,
        "token_usage": usage,
    }
