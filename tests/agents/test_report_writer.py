# tests/agents/test_report_writer.py
import pytest
from unittest.mock import patch
from src.agents.report_writer import ReportWriterAgent
from src.models.schemas import ScoredOpportunity

OPPORTUNITY = ScoredOpportunity(
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
    current_fit_score=8.5,
    reachable_fit_score=9.0,
    fit_rationale="Strong fraud ML background",
    relevant_skills=["XGBoost"],
    skills_to_learn=["sales"],
    learning_effort="low",
)

MARKDOWN = "# Business Opportunity Report\n\n## Top Opportunities\n..."


def test_returns_string():
    with patch.object(ReportWriterAgent, '_cli_call', return_value=MARKDOWN):
        agent = ReportWriterAgent()
        result = agent.run([OPPORTUNITY], "")
    assert isinstance(result, str)


def test_returns_markdown_content():
    with patch.object(ReportWriterAgent, '_cli_call', return_value=MARKDOWN):
        agent = ReportWriterAgent()
        result = agent.run([OPPORTUNITY], "")
    assert "Business Opportunity Report" in result


def test_incorporates_critique():
    captured = []

    def fake_cli(system, user):
        captured.append(user)
        return MARKDOWN

    with patch.object(ReportWriterAgent, '_cli_call', side_effect=fake_cli):
        agent = ReportWriterAgent()
        agent.run([OPPORTUNITY], "TAM estimate lacks citation")

    assert any("TAM estimate lacks citation" in u for u in captured)
