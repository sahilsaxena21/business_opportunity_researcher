# tests/agents/test_skill_fit_scorer.py
import json
import pytest
from unittest.mock import patch
from src.agents.skill_fit_scorer import SkillFitScorerAgent
from src.models.schemas import GapAnalyzedOpportunity, ScoredOpportunity

CANDIDATE = GapAnalyzedOpportunity(
    title="AI Fraud Detection",
    description="ML-based fraud",
    industry="fintech",
    sources=["McKinsey"],
    tam_estimate="$4.2B",
    tam_math="5000 banks × $840k",
    tam_passes=True,
    gap_summary="Legacy systems miss novel attacks",
    existing_solutions=["Featurespace"],
    key_gap="Real-time adaptive ML",
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k",
    "tam_passes": True,
    "gap_summary": "Legacy systems miss novel attacks",
    "existing_solutions": ["Featurespace"],
    "key_gap": "Real-time adaptive ML",
    "current_fit_score": 8.5,
    "reachable_fit_score": 9.0,
    "fit_rationale": "Strong fraud ML background",
    "relevant_skills": ["XGBoost", "behavioral ML"],
    "skills_to_learn": ["sales"],
    "learning_effort": "low",
}])

PROFILE = {"raw": "Data scientist with fraud detection experience"}


def test_returns_list():
    with patch.object(SkillFitScorerAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = SkillFitScorerAgent()
        results = agent.run([CANDIDATE], PROFILE)
    assert isinstance(results, list)


def test_returns_scored_opportunities():
    with patch.object(SkillFitScorerAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = SkillFitScorerAgent()
        results = agent.run([CANDIDATE], PROFILE)
    assert all(isinstance(r, ScoredOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = SkillFitScorerAgent()
    assert agent.run([], PROFILE) == []
