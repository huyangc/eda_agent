from app.agent.state import AgentState
from app.config import settings


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "ambiguous")
    confidence = state.get("intent_confidence", 0.0)

    if intent == "general":
        return "passthrough"
    if intent == "eda" and confidence >= settings.intent_confidence_threshold:
        return "command_extractor"
    # ambiguous or low-confidence EDA → still try the EDA path
    return "command_extractor"


def route_after_command_extractor(state: AgentState) -> str:
    namespace = state.get("detected_tool_namespace", "none")
    if namespace == "none":
        return "passthrough"
    return "rag_retriever"
