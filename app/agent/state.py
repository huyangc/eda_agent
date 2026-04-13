import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Input — set once at graph entry
    messages: Annotated[list[BaseMessage], operator.add]
    request_id: str
    forced_tool: Optional[str]   # "primetime" | "innovus" | "both" | None
    forced_mode: Optional[str]   # "qa" | "tcl" | None
    # Anthropic-format tool definitions forwarded from the client request.
    # Used only by passthrough_node; EDA path ignores tools entirely.
    tools: list[dict]
    # Structured content blocks for tool_use responses (None = plain text)
    response_content_blocks: Optional[list[dict]]

    # Intent detection
    intent: Optional[str]        # "eda" | "general" | "ambiguous"
    intent_confidence: float

    # Command / namespace analysis
    detected_tool_namespace: Optional[str]  # "primetime" | "innovus" | "both" | "none"
    raw_command_candidates: list[str]

    # RAG results
    retrieved_docs: list[dict]   # [{content, metadata, score}]
    retrieval_summary: Optional[str]

    # Output
    output_mode: str             # "qa" | "tcl" | "unknown"
    final_response: Optional[str]
    token_usage: dict

    # Streaming — asyncio.Queue injected by the API layer; None for sync calls
    stream_queue: Optional[object]

    # Tracing — TraceWriter injected by the API layer
    trace_writer: Optional[object]
