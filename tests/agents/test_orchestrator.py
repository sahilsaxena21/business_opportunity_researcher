import pytest
from unittest.mock import MagicMock
from src.agents.orchestrator import Orchestrator, merge_candidates
from src.models.schemas import TrendCandidate, SignalCandidate, OpportunityCandidate


def test_merge_combines_trends_and_signals():
    trends = [
        TrendCandidate(title="AI Fraud Detection", description="d", industry="fintech", source="McKinsey"),
        TrendCandidate(title="Supply Chain AI", description="d2", industry="retail", source="BCG"),
    ]
    signals = [
        SignalCandidate(title="VC surge in IoT", description="d3", signal_type="vc_investment", source="TC", strength="strong"),
    ]
    result = merge_candidates(trends, signals)
    assert len(result) == 3
    assert all(isinstance(c, OpportunityCandidate) for c in result)


def test_merge_deduplicates_overlapping():
    trends = [TrendCandidate(title="AI fraud detection tools", description="d", industry="fintech", source="McKinsey")]
    signals = [SignalCandidate(title="AI fraud detection surge", description="VC money", signal_type="vc_investment", source="TC", strength="strong")]
    result = merge_candidates(trends, signals)
    assert len(result) == 1


def test_merge_empty_inputs():
    assert merge_candidates([], []) == []


def test_orchestrator_initializes():
    orch = Orchestrator(anthropic_client=MagicMock(), tavily_client=MagicMock())
    assert orch is not None
