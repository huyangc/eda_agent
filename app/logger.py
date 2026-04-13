"""Centralized logging configuration for EDA Agent.

Logs are written to both stdout and a date-partitioned file:
    logs/
    └── 2026-04-13/
        └── eda_agent.log

Log format shows a clear per-request trace:
    [req-abc12] ── NEW REQUEST ── POST /v1/messages  user: "What does report_timing..."
    [req-abc12] [intent_detector ]  eda  (confidence=0.95)
    [req-abc12] [cmd_extractor   ]  namespace=primetime  commands=['report_timing', ...]  mode=qa
    [req-abc12] [rag_retriever   ]  RAG ON  queries=2  docs_retrieved=5  top_score=0.94
    [req-abc12] [response_gen    ]  mode=qa  streaming  chars=842
    [req-abc12] ── DONE ── 4.21s
"""

import logging
import sys
from datetime import date
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    # date-partitioned file handler: logs/YYYY-MM-DD/eda_agent.log
    today = date.today().isoformat()
    log_path = Path(log_dir) / today
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "eda_agent.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "openai", "langchain", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def req_id(request_id: str) -> str:
    """Return a short, bracketed request ID prefix for log lines."""
    short = request_id[-8:] if len(request_id) > 8 else request_id
    return f"[{short}]"
