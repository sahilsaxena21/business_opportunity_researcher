# tests/test_schemas.py
import pytest
from datetime import datetime
from src.models.schemas import (
    TrendCandidate, SignalCandidate, OpportunityCandidate,
    SizedOpportunity, GapAnalyzedOpportunity, ScoredOpportunity,
    OpportunityBrief, CritiqueItem, CritiqueResult, FinalReport,
)


def test_trend_candidate_valid():
    t = TrendCandidate(title="AI in fraud", description="desc", industry="fintech", source="McKinsey")
    assert t.title == "AI in fraud"


def test_sized_opportunity_inherits_candidate():
    o = SizedOpportunity(
        title="Fraud AI", description="desc", industry="fintech", sources=["McKinsey"],
        tam_estimate="$2.4B", tam_math="48000 x $50000 = $2.4B", tam_passes=True,
    )
    assert o.tam_passes is True


def test_scored_opportunity_fit_score_range():
    o = ScoredOpportunity(
        title="Fraud AI", description="desc", industry="fintech", sources=["McKinsey"],
        tam_estimate="$2.4B", tam_math="48000 x $50000 = $2.4B", tam_passes=True,
        gap_summary="no ML", existing_solutions=["ReedCase"], key_gap="no behavioral ML",
        current_fit_score=8.5, reachable_fit_score=9.0,
        fit_rationale="matches XGBoost fraud work; content marketing learnable",
        relevant_skills=["XGBoost"], skills_to_learn=["content marketing"],
        learning_effort="low",
    )
    assert 0.0 <= o.current_fit_score <= 10.0
    assert 0.0 <= o.reachable_fit_score <= 10.0


def test_critique_result_structure():
    r = CritiqueResult(
        round_number=1,
        has_changes=True,
        critiques=[CritiqueItem(
            opportunity_title="Fraud AI",
            issue="TAM math unverified",
            suggested_improvement="Add source for 48000 figure",
        )],
        overall_quality="fair",
    )
    assert r.round_number == 1
    assert len(r.critiques) == 1


def test_final_report_requires_markdown():
    r = FinalReport(
        generated_at=datetime.now(),
        total_opportunities_found=5,
        ranked_opportunities=[],
        critique_rounds=[],
        report_markdown="# Report",
    )
    assert r.report_markdown == "# Report"
