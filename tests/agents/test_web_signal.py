# tests/agents/test_web_signal.py
import pytest
from unittest.mock import MagicMock
from src.agents.web_signal import WebSignalAgent
from src.models.schemas import SignalCandidate


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {"title": "VC surge in AI fraud tools", "description": "5 rounds >$50M in 2025", "signal_type": "vc_investment", "source": "TechCrunch", "strength": "strong"},
  {"title": "Job spike: ML fraud engineers", "description": "300% YoY increase", "signal_type": "job_spike", "source": "LinkedIn", "strength": "strong"}
]'''


def test_returns_list():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("fraud ML expert", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_signal_candidates():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("fraud ML expert", lambda q: "[]")
    assert all(isinstance(r, SignalCandidate) for r in results)


def test_parses_signal_type():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("profile", lambda q: "[]")
    assert any(r.signal_type == "vc_investment" for r in results)


def test_returns_empty_on_bad_json():
    agent = WebSignalAgent(make_mock_client("No signals found."))
    results = agent.run("profile", lambda q: "[]")
    assert results == []
