"""Mock retriever: calls OpenAI to generate plausible EDA documentation chunks."""

import json
import logging

from openai import AsyncOpenAI

from rag_service.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a technical documentation generator for EDA (Electronic Design Automation) tools.

Given a search query and a tool namespace, generate {top_k} realistic documentation snippets \
that would be retrieved from a vector database of EDA command reference manuals.

Each snippet should look like an excerpt from actual {namespace} documentation — \
covering command syntax, options/flags, usage notes, or example TCL code as appropriate \
for the query topic.

Respond with a JSON array of exactly {top_k} objects, each with these fields:
- "content": the documentation text (2–6 sentences or a short code example, ~150-300 chars)
- "command_name": the primary EDA command name this chunk is about (snake_case or camelCase)
- "source_type": one of "man_page", "user_guide", "tcl_help"
- "score": a relevance score between 0.75 and 0.99 (float, varies across chunks)

Only output the JSON array — no markdown fences, no prose."""

_NAMESPACE_CONTEXT = {
    "primetime": "Synopsys PrimeTime (Static Timing Analysis). Commands: report_timing, "
                 "report_constraint, get_cells, get_nets, get_pins, get_timing_paths, "
                 "read_sdc, read_verilog, link_design, check_timing, set_max_delay, "
                 "set_false_path, set_multicycle_path, report_clock, etc.",
    "innovus":   "Cadence Innovus (Place & Route). Commands: place_design, route_design, "
                 "floorPlan, add_fillers, clockDesign, setDesignMode, optDesign, "
                 "add_io_fillers, saveDesign, reportDesignStat, routeDesign, etc.",
    "both":      "Synopsys PrimeTime (STA) and Cadence Innovus (Place & Route).",
}


async def generate_mock_docs(
    query: str,
    namespace: str,
    top_k: int,
) -> list[dict]:
    """Ask the LLM to produce top_k fake-but-plausible EDA doc chunks."""
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    ns_context = _NAMESPACE_CONTEXT.get(namespace, namespace)
    system = _SYSTEM_PROMPT.format(top_k=top_k, namespace=ns_context)

    try:
        response = await client.chat.completions.create(
            model=settings.rag_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Query: {query}\nNamespace: {namespace}"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
        # The model might wrap the array in an object key
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Try common wrapper keys
            for key in ("docs", "results", "chunks", "data"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                # Fall back to first list value found
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = []
    except Exception as exc:
        logger.warning("LLM retriever error: %s", exc)
        return []

    docs = []
    for item in parsed[:top_k]:
        if not isinstance(item, dict):
            continue
        docs.append({
            "content": str(item.get("content", "")),
            "metadata": {
                "tool": namespace if namespace != "both" else item.get("tool", "primetime"),
                "command_name": item.get("command_name", ""),
                "source_type": item.get("source_type", "man_page"),
                "doc_version": "mock-v1",
            },
            "score": float(item.get("score", 0.85)),
        })

    return docs
