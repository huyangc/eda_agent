"""Session-level trace writer.

Each request gets a JSONL file under:
    logs/YYYY-MM-DD/{HHMMSS}_{session_id}.jsonl

Every line is a self-contained JSON event:
    {"ts": "09:40:34.123", "type": "request",      "data": { full incoming messages }}
    {"ts": "09:40:53.210", "type": "intent",        "data": { intent, confidence, reason }}
    {"ts": "09:41:15.441", "type": "cmd_extract",   "data": { namespace, commands, mode }}
    {"ts": "09:41:15.890", "type": "rag_query",     "data": { query, namespace, top_k }}
    {"ts": "09:41:16.320", "type": "rag_response",  "data": { docs: [{content, metadata, score}] }}
    {"ts": "09:42:07.001", "type": "llm_response",  "data": { mode, content }}
"""

import json
import threading
from datetime import date, datetime
from pathlib import Path


class TraceWriter:
    """Writes structured JSONL trace events for a single request/session."""

    _lock = threading.Lock()

    def __init__(self, session_id: str, log_dir: str = "logs") -> None:
        self.session_id = session_id
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        time_prefix = now.strftime("%H%M%S")
        dir_path = Path(log_dir) / today
        dir_path.mkdir(parents=True, exist_ok=True)
        self._path = dir_path / f"{time_prefix}_{session_id}.jsonl"

    def write(self, event_type: str, data: dict) -> None:
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:12],
            "session_id": self.session_id,
            "type": event_type,
            "data": data,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ---- Convenience methods for each event type ---------------------------

    def request(self, system: str | None, messages: list[dict]) -> None:
        self.write("request", {
            "system": system,
            "messages": messages,
        })

    def intent(self, intent: str, confidence: float, reason: str) -> None:
        self.write("intent", {
            "intent": intent,
            "confidence": confidence,
            "reason": reason,
        })

    def cmd_extract(self, namespace: str, commands: list, mode: str) -> None:
        self.write("cmd_extract", {
            "namespace": namespace,
            "commands": commands,
            "output_mode": mode,
        })

    def rag_query(self, query: str, namespace: str, top_k: int) -> None:
        self.write("rag_query", {
            "query": query,
            "namespace": namespace,
            "top_k": top_k,
        })

    def rag_response(self, query: str, docs: list[dict]) -> None:
        self.write("rag_response", {
            "query": query,
            "docs": docs,
        })

    def llm_response(self, mode: str, content: str) -> None:
        self.write("llm_response", {
            "output_mode": mode,
            "content": content,
        })

    def tool_calls(self, calls: list[dict]) -> None:
        """LLM decided to call tools — log what it asked for."""
        self.write("tool_calls", {"calls": calls})

    def tool_results(self, results: list[dict]) -> None:
        """Tool execution results returned to the LLM."""
        self.write("tool_results", {"results": results})
