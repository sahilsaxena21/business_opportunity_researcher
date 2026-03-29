import pytest
from unittest.mock import MagicMock
from src.agents.skill_fit_scorer import SkillFitScorerAgent
from src.models.schemas import GapAnalyzedOpportunity, ScoredOpportunity


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
    "gap_summary": "No behavioral ML", "existing_solutions": ["NICE Actimize"], "key_gap": "No real-time features",
    "current_fit_score": 9.0,
    "reachable_fit_score": 9.5,
    "fit_rationale": "Sahil built fraud detection at Scotiabank using XGBoost; content marketing learnable in months",
    "relevant_skills": ["XGBoost", "fraud detection", "clickstream analysis"],
    "skills_to_learn": ["content marketing"],
    "learning_effort": "low"
  }
]'''

CANDIDATES = [
    GapAnalyzedOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
        gap_summary="No behavioral ML", existing_solutions=["NICE Actimize"], key_gap="No real-time features",
    )
]

PROFILE = {"name": "Sahil", "raw": "XGBoost fraud detection at Scotiabank, agentic AI pipelines"}


def test_returns_list():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES, PROFILE), list)


def test_returns_scored_opportunities():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert all(isinstance(r, ScoredOpportunity) for r in results)


def test_scores_in_range():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert 0.0 <= results[0].current_fit_score <= 10.0
    assert 0.0 <= results[0].reachable_fit_score <= 10.0


def test_learning_effort_valid_value():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert results[0].learning_effort in ("low", "medium", "high")


def test_skills_to_learn_is_list():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert isinstance(results[0].skills_to_learn, list)
