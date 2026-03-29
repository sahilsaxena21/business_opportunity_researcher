# tests/agents/test_consulting_trend.py
import pytest
from unittest.mock import patch
from src.agents.consulting_trend import ConsultingTrendAgent
from src.models.schemas import TrendCandidate

SAMPLE_JSON = '''[
  {"title": "AI-Driven Fraud Detection", "description": "Banks adopting ML", "industry": "fintech", "source": "McKinsey"},
  {"title": "Supply Chain AI", "description": "AI optimizing logistics", "industry": "retail", "source": "BCG"}
]'''


def test_returns_list():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("Data scientist profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_trend_candidates():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("Data scientist profile", lambda q: "[]")
    assert all(isinstance(r, TrendCandidate) for r in results)


def test_parses_title():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "AI-Driven Fraud Detection" for r in results)


def test_returns_empty_on_bad_json():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value="Sorry, no trends found."):
        agent = ConsultingTrendAgent()
        results = agent.run("profile", lambda q: "[]")
    assert results == []
