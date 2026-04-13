import hashlib

from app.agent.state import AgentState
from app.config import settings
from app.rag.client import RAGClient
from app.logger import get_logger, req_id

logger = get_logger(__name__)


def _last_user_content(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return str(msg.content)
    return ""


def _doc_fingerprint(doc: dict) -> str:
    return hashlib.md5(doc.get("content", "").encode()).hexdigest()


def _format_retrieval_summary(docs: list[dict]) -> str:
    if not docs:
        return "(No relevant documentation retrieved.)"
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        tool = meta.get("tool", "")
        cmd = meta.get("command_name", "")
        header = f"[{i}]"
        if tool:
            header += f" [{tool.upper()}]"
        if cmd:
            header += f" {cmd}"
        parts.append(f"{header}\n{doc['content'].strip()}")
    return "\n\n---\n\n".join(parts)


async def rag_retriever_node(state: AgentState) -> dict:
    """Retrieve relevant EDA documentation from the external RAG service."""
    client = RAGClient.get_instance()
    namespace = state["detected_tool_namespace"]
    candidates = state["raw_command_candidates"]
    user_query = _last_user_content(state["messages"])

    # Issue one query per command candidate for better precision,
    # each enriched with the user's original query for context.
    k_per_query = max(1, settings.rag_top_k // max(len(candidates), 1))
    seen: dict[str, dict] = {}

    for candidate in candidates:
        query = f"{candidate} {user_query}".strip()
        docs = await client.retrieve(query=query, namespace=namespace, top_k=k_per_query)
        for doc in docs:
            fp = _doc_fingerprint(doc)
            if fp not in seen:
                seen[fp] = doc

    # If no candidates extracted, fall back to a single query with the raw user text
    if not candidates:
        docs = await client.retrieve(query=user_query, namespace=namespace)
        for doc in docs:
            fp = _doc_fingerprint(doc)
            if fp not in seen:
                seen[fp] = doc

    # Rank by score, take top K
    all_docs = sorted(seen.values(), key=lambda d: d.get("score", 0.0), reverse=True)
    top_docs = all_docs[:settings.rag_top_k]

    # Filter below score threshold
    top_docs = [d for d in top_docs if d.get("score", 1.0) >= settings.rag_score_threshold]

    rid = req_id(state.get("request_id", ""))
    top_score = max((d.get("score", 0.0) for d in top_docs), default=0.0)
    logger.info(
        "%s [rag_retriever    ]  RAG %-3s  namespace=%-12s  queries=%d  docs=%d  top_score=%.2f",
        rid,
        "ON" if top_docs else "ON(0 docs)",
        namespace,
        len(candidates) or 1,
        len(top_docs),
        top_score,
    )
    return {
        "retrieved_docs": top_docs,
        "retrieval_summary": _format_retrieval_summary(top_docs),
    }
