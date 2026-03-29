# tests/agents/test_critic.py
import json
from unittest.mock import patch
from src.agents.critic import CriticAgent
from src.models.schemas import CritiqueResult

GOOD_JSON = json.dumps({
    "round_number": 1,
    "has_changes": True,
    "critiques": [{"opportunity_title": "AI Fraud", "issue": "TAM unsupported", "suggested_improvement": "Cite source"}],
    "overall_quality": "fair",
})

NO_CHANGE_JSON = json.dumps({
    "round_number": 2,
    "has_changes": False,
    "critiques": [],
    "overall_quality": "excellent",
})


def test_returns_critique_result():
    with patch.object(CriticAgent, '_cli_call', return_value=GOOD_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert isinstance(result, CritiqueResult)


def test_parses_has_changes():
    with patch.object(CriticAgent, '_cli_call', return_value=GOOD_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert result.has_changes is True


def test_parses_no_changes():
    with patch.object(CriticAgent, '_cli_call', return_value=NO_CHANGE_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=2)
    assert result.has_changes is False
    assert result.critiques == []


def test_returns_default_on_bad_json():
    with patch.object(CriticAgent, '_cli_call', return_value="not json"):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert isinstance(result, CritiqueResult)
    assert result.has_changes is False
