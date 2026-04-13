import json

from langchain_core.messages import BaseMessage

from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.logger import get_logger, req_id

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for an EDA (Electronic Design Automation) assistant.

Classify the user's latest message into exactly one of:
- "eda": the request is about EDA tools, timing analysis, place-and-route, TCL scripting,
         Primetime, Innovus, constraints, SDC, netlist, floorplan, clock tree, etc.
- "general": the request is a general programming question, greeting, or unrelated topic.
- "ambiguous": insufficient context to determine.

Respond with JSON only — no prose, no markdown:
{"intent": "<label>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}"""


def _last_user_content(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return str(msg.content)
    return ""


async def intent_detector_node(state: AgentState) -> dict:
    """Classify whether the user's request is EDA-related."""
    user_text = _last_user_content(state["messages"])

    llm = get_llm(json_mode=True, temperature=0.0)
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ])

    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "ambiguous")
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):
        logger.warning("intent_detector: failed to parse JSON: %s", response.content)
        intent = "ambiguous"
        confidence = 0.5

    rid = req_id(state.get("request_id", ""))
    reason = parsed.get("reason", "")
    logger.info(
        "%s [intent_detector   ]  intent=%-10s  confidence=%.2f  reason=%s",
        rid, intent, confidence, reason,
    )
    if tw := state.get("trace_writer"):
        tw.intent(intent, confidence, reason)
    return {"intent": intent, "intent_confidence": confidence}
