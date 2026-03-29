# tests/agents/test_consulting_trend.py
import pytest
from unittest.mock import MagicMock
from src.agents.consulting_trend import ConsultingTrendAgent
from src.models.schemas import TrendCandidate


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {"title": "AI-Driven Fraud Detection", "description": "Banks adopting ML", "industry": "fintech", "source": "McKinsey"},
  {"title": "Supply Chain AI", "description": "AI optimizing logistics", "industry": "retail", "source": "BCG"}
]'''


def test_returns_list():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("Data scientist profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_trend_candidates():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("Data scientist profile", lambda q: "[]")
    assert all(isinstance(r, TrendCandidate) for r in results)


def test_parses_title():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "AI-Driven Fraud Detection" for r in results)


def test_returns_empty_on_bad_json():
    agent = ConsultingTrendAgent(make_mock_client("Sorry, no trends found."))
    results = agent.run("profile", lambda q: "[]")
    assert results == []
