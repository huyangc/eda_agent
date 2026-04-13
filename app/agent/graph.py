from langgraph.graph import END, StateGraph

from app.agent.nodes.command_extractor import command_extractor_node
from app.agent.nodes.intent_detector import intent_detector_node
from app.agent.nodes.passthrough import passthrough_node
from app.agent.nodes.rag_retriever import rag_retriever_node
from app.agent.nodes.response_generator import response_generator_node
from app.agent.routing import route_after_command_extractor, route_after_intent
from app.agent.state import AgentState


def _route_entry(state: AgentState) -> str:
    """Short-circuit to passthrough when the request carries tools.

    Tool-bearing requests come from Claude Code subagents performing file /
    shell operations — they are never EDA queries and must never enter the
    intent / RAG pipeline.
    """
    if state.get("tools") or state.get("forced_tool"):
        # forced_tool is set only when the API caller explicitly picks an EDA
        # namespace, so only plain tools (subagent calls) get the short-circuit.
        if state.get("tools") and not state.get("forced_tool"):
            return "passthrough"
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
