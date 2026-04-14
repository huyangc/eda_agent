from langgraph.graph import END, StateGraph

from app.agent.nodes.command_extractor import command_extractor_node
from app.agent.nodes.intent_detector import intent_detector_node
from app.agent.nodes.passthrough import passthrough_node
from app.agent.nodes.rag_retriever import rag_retriever_node
from app.agent.nodes.response_generator import response_generator_node
from app.agent.routing import route_after_command_extractor, route_after_intent
from app.agent.state import AgentState


def _route_entry(state: AgentState) -> str:
    """Intelligently route requests at graph entry.

    - Forced tool via API always bypasses short-circuits and enters intent.
    - If it's an ongoing tool loop (subagent returning tool output), short-circuit to passthrough.
    - Otherwise (real human query with text), enter intent_detector, even if tools are present.
    """
    if state.get("forced_tool") and state.get("forced_tool") != "auto":
        return "intent_detector"

    messages = state["messages"]
    if not messages:
        return "passthrough"

    last_msg = messages[-1]

    # If the last message is a ToolMessage (tool execution result) -> ongoing tool loop
    if last_msg.type == "tool":
        return "passthrough"
        
    # If the last message is a HumanMessage but empty or whitespace -> ongoing tool loop
    if last_msg.type == "human" and not str(last_msg.content).strip():
        return "passthrough"

    # Human message with real text -> must detect intent
    return "intent_detector"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intent_detector", intent_detector_node)
    g.add_node("command_extractor", command_extractor_node)
    g.add_node("rag_retriever", rag_retriever_node)
    g.add_node("response_generator", response_generator_node)
    g.add_node("passthrough", passthrough_node)

    g.set_conditional_entry_point(
        _route_entry,
        {"intent_detector": "intent_detector", "passthrough": "passthrough"},
    )

    g.add_conditional_edges(
        "intent_detector",
        route_after_intent,
        {"command_extractor": "command_extractor", "passthrough": "passthrough"},
    )

    g.add_conditional_edges(
        "command_extractor",
        route_after_command_extractor,
        {"rag_retriever": "rag_retriever", "passthrough": "passthrough"},
    )

    g.add_edge("rag_retriever", "response_generator")
    g.add_edge("response_generator", END)
    g.add_edge("passthrough", END)

    return g.compile()


# Compiled once at import time; reused for every request.
eda_graph = build_graph()
