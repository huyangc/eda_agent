import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.config import settings
from app.llm.factory import get_llm
from app.logger import get_logger, req_id

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are analyzing a user request for an EDA assistant.

Known tool namespaces:
- "primetime": Synopsys PrimeTime (Static Timing Analysis). Commands include
  report_timing, report_constraint, get_cells, get_nets, get_pins, read_sdc,
  read_verilog, link_design, check_timing, etc.
- "innovus": Cadence Innovus (Place & Route). Commands include place_design,
  route_design, floorPlan, add_fillers, clockDesign, setDesignMode, etc.

Tasks:
1. Determine which namespace(s) apply: "primetime", "innovus", "both", or "none".
2. Extract up to {max_cmds} specific EDA command names OR topic keywords to use
   as RAG search queries. Prefer exact command names when mentioned; otherwise
   use the operation concept (e.g. "setup slack", "clock skew", "hold violation").
3. Determine the output mode: "tcl" if the user wants code/a script; "qa" otherwise.

Respond with JSON only:
{{"namespace": "<primetime|innovus|both|none>", "commands": ["cmd1", ...], "output_mode": "<qa|tcl>"}}"""


def _last_user_content(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return str(msg.content)
    return ""


async def command_extractor_node(state: AgentState) -> dict:
    """Extract EDA tool namespace, command candidates, and output mode."""
    user_text = _last_user_content(state["messages"])

    system = _SYSTEM_PROMPT.format(max_cmds=settings.max_commands_to_retrieve)
    llm = get_llm(json_mode=True, temperature=0.0)

    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_text),
    ])

    try:
        parsed = json.loads(response.content)
        namespace = parsed.get("namespace", "none")
        commands = parsed.get("commands", [])[:settings.max_commands_to_retrieve]
        output_mode = parsed.get("output_mode", "qa")
    except (json.JSONDecodeError, ValueError):
        logger.warning("command_extractor: failed to parse JSON: %s", response.content)
        namespace = "none"
        commands = []
        output_mode = "qa"

    # Honour overrides from the API request
    if state.get("forced_tool") and state["forced_tool"] != "auto":
        namespace = state["forced_tool"]
    if state.get("forced_mode") and state["forced_mode"] != "auto":
        output_mode = state["forced_mode"]

    rid = req_id(state.get("request_id", ""))
    logger.info(
        "%s [cmd_extractor    ]  namespace=%-12s  commands=%s  mode=%s",
        rid, namespace, commands, output_mode,
    )
    return {
        "detected_tool_namespace": namespace,
        "raw_command_candidates": commands,
        "output_mode": output_mode,
    }
