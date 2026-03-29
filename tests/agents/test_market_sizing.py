import pytest
from unittest.mock import MagicMock
from src.agents.market_sizing import MarketSizingAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {
    "title": "AI Fraud Detection SaaS",
    "description": "ML platform for bank fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$3.2B",
    "tam_math": "6000 banks x $53000/yr = $3.2B",
    "tam_passes": true
  }
]'''

CANDIDATES = [
    OpportunityCandidate(
        title="AI Fraud Detection SaaS",
        description="ML platform for bank fraud",
        industry="fintech",
        sources=["McKinsey"],
    )
]


def test_returns_list():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES), list)


def test_returns_sized_opportunities():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert all(isinstance(r, SizedOpportunity) for r in results)


def test_preserves_title():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert agent.run(CANDIDATES)[0].title == "AI Fraud Detection SaaS"


def test_sets_tam_passes():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert agent.run(CANDIDATES)[0].tam_passes is True


def test_empty_candidates_returns_empty():
    agent = MarketSizingAgent(make_mock_client("[]"))
    assert agent.run([]) == []
