# tests/agents/test_market_sizing.py
import json
import pytest
from unittest.mock import patch
from src.agents.market_sizing import MarketSizingAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity

CANDIDATE = OpportunityCandidate(
    title="AI Fraud Detection",
    description="ML-based fraud platform",
    industry="fintech",
    sources=["McKinsey"],
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud platform",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k/yr",
    "tam_passes": True,
}])


def test_returns_list():
    with patch.object(MarketSizingAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert isinstance(results, list)


def test_returns_sized_opportunities():
    with patch.object(MarketSizingAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert all(isinstance(r, SizedOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = MarketSizingAgent()
    assert agent.run([]) == []


def test_returns_empty_on_bad_json():
    with patch.object(MarketSizingAgent, '_cli_call', return_value="not json"):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert results == []
