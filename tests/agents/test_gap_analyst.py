# tests/agents/test_gap_analyst.py
import json
import pytest
from unittest.mock import patch
from src.agents.gap_analyst import GapAnalystAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity

CANDIDATE = SizedOpportunity(
    title="AI Fraud Detection",
    description="ML-based fraud",
    industry="fintech",
    sources=["McKinsey"],
    tam_estimate="$4.2B",
    tam_math="5000 banks × $840k",
    tam_passes=True,
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k",
    "tam_passes": True,
    "gap_summary": "Legacy rules-based systems miss novel attacks",
    "existing_solutions": ["Featurespace", "Sardine"],
    "key_gap": "Real-time adaptive ML for novel fraud patterns",
}])


def test_returns_list():
    with patch.object(GapAnalystAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = GapAnalystAgent()
        results = agent.run([CANDIDATE])
    assert isinstance(results, list)


def test_returns_gap_analyzed():
    with patch.object(GapAnalystAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = GapAnalystAgent()
        results = agent.run([CANDIDATE])
    assert all(isinstance(r, GapAnalyzedOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = GapAnalystAgent()
    assert agent.run([]) == []
