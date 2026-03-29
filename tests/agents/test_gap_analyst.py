import pytest
from unittest.mock import MagicMock
from src.agents.gap_analyst import GapAnalystAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {
    "title": "AI Fraud Detection SaaS", "description": "ML platform", "industry": "fintech",
    "sources": ["McKinsey"], "tam_estimate": "$3.2B", "tam_math": "6000 x $53000 = $3.2B", "tam_passes": true,
    "gap_summary": "Existing tools lack behavioral ML signals",
    "existing_solutions": ["NICE Actimize", "SAS Fraud"],
    "key_gap": "No real-time behavioral feature engineering"
  }
]'''

CANDIDATES = [
    SizedOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
    )
]


def test_returns_list():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES), list)


def test_returns_gap_analyzed_opportunities():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert all(isinstance(r, GapAnalyzedOpportunity) for r in results)


def test_sets_key_gap():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert results[0].key_gap == "No real-time behavioral feature engineering"


def test_sets_existing_solutions():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert "NICE Actimize" in results[0].existing_solutions
