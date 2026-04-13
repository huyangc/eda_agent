"""Basic API smoke tests (no LLM/RAG calls — uses mocks)."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

app = create_app()
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "eda-agent" in ids


@patch("app.agent.nodes.intent_detector.get_llm")
@patch("app.agent.nodes.command_extractor.get_llm")
@patch("app.agent.nodes.response_generator.get_llm")
@patch("app.rag.client.RAGClient.get_instance")
def test_chat_completion_non_streaming(
    mock_rag_instance,
    mock_gen_llm,
    mock_cmd_llm,
    mock_intent_llm,
):
    # Intent: EDA
    intent_mock = AsyncMock()
    intent_mock.ainvoke = AsyncMock(
        return_value=type("R", (), {"content": '{"intent":"eda","confidence":0.95,"reason":"timing"}'})()
    )
    mock_intent_llm.return_value = intent_mock

    # Command extraction
    cmd_mock = AsyncMock()
    cmd_mock.ainvoke = AsyncMock(
        return_value=type("R", (), {
            "content": '{"namespace":"primetime","commands":["report_timing"],"output_mode":"qa"}'
        })()
    )
    mock_cmd_llm.return_value = cmd_mock

    # RAG client
    rag_client = AsyncMock()
    rag_client.retrieve = AsyncMock(return_value=[
        {"content": "report_timing prints the timing path.", "metadata": {"tool": "primetime"}, "score": 0.9}
    ])
    mock_rag_instance.return_value = rag_client

    # Response generator
    async def fake_astream(messages):
        for token in ["Hello", " world"]:
            yield type("C", (), {"content": token})()

    gen_mock = MagicMock()
    gen_mock.astream = fake_astream
    mock_gen_llm.return_value = gen_mock

    resp = client.post("/v1/chat/completions", json={
        "model": "eda-agent",
        "messages": [{"role": "user", "content": "What does report_timing do?"}],
        "stream": False,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"


# needed for the mock above
from unittest.mock import MagicMock  # noqa: E402
