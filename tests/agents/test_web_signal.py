# tests/agents/test_web_signal.py
from unittest.mock import patch
from src.agents.web_signal import WebSignalAgent
from src.models.schemas import SignalCandidate

SAMPLE_JSON = '''[
  {"title": "VC surge in fraud AI", "description": "Heavy investment", "signal_type": "vc_investment", "source": "TechCrunch", "strength": "strong"},
  {"title": "Job spike in MLOps", "description": "Growing demand", "signal_type": "job_spike", "source": "LinkedIn", "strength": "moderate"}
]'''


def test_returns_list():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_signal_candidates():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert all(isinstance(r, SignalCandidate) for r in results)


def test_parses_title():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "VC surge in fraud AI" for r in results)


def test_returns_empty_on_bad_json():
    with patch.object(WebSignalAgent, '_cli_call', return_value="No signals."):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert results == []
