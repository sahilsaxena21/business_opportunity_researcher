import pytest
from unittest.mock import MagicMock
from src.agents.critic import CriticAgent
from src.models.schemas import CritiqueResult


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


WITH_CHANGES = '''{
  "round_number": 1,
  "has_changes": true,
  "critiques": [
    {"opportunity_title": "AI Fraud SaaS", "issue": "TAM lacks source", "suggested_improvement": "Cite FDIC data"}
  ],
  "overall_quality": "fair"
}'''

NO_CHANGES = '''{
  "round_number": 2,
  "has_changes": false,
  "critiques": [],
  "overall_quality": "good"
}'''


def test_returns_critique_result():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert isinstance(agent.run("# Report", round_number=1), CritiqueResult)


def test_parses_has_changes_true():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert agent.run("# Report", round_number=1).has_changes is True


def test_parses_critiques_list():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    result = agent.run("# Report", round_number=1)
    assert len(result.critiques) == 1
    assert result.critiques[0].opportunity_title == "AI Fraud SaaS"


def test_no_changes_returns_empty_critiques():
    agent = CriticAgent(make_mock_client(NO_CHANGES))
    result = agent.run("# Good report", round_number=2)
    assert result.has_changes is False
    assert result.critiques == []


def test_sets_round_number():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert agent.run("# Report", round_number=1).round_number == 1
