import pytest
from unittest.mock import MagicMock
from src.agents.report_writer import ReportWriterAgent
from src.models.schemas import ScoredOpportunity

SAMPLE_REPORT = """# Business Opportunity Report

## Ranked Shortlist

| Rank | Opportunity | Industry | TAM | Skill Fit |
|---|---|---|---|---|
| 1 | AI Fraud Detection SaaS | fintech | $3.2B | 9.0 |

## Deep-Dive: AI Fraud Detection SaaS

### The Trend
Banks are rapidly adopting ML-based fraud detection. Source: McKinsey.

### The Gap
No real-time behavioral feature engineering in existing tools.

### Market Size
6000 banks × $53000/yr = $3.2B

### Your Edge
Sahil built XGBoost fraud detection at Scotiabank using clickstream data.

### Recommended Business Model
B2B SaaS with annual subscription. 80%+ margins at scale.

### Confidence Score
8.5/10 — strong TAM, clear gap, excellent founder fit.
"""


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


# CORRECTED: use current_fit_score/reachable_fit_score (not fit_score), plus required fields
SCORED = [
    ScoredOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
        gap_summary="No behavioral ML", existing_solutions=["NICE Actimize"], key_gap="No real-time features",
        current_fit_score=9.0, reachable_fit_score=9.5,
        fit_rationale="Built at Scotiabank", relevant_skills=["XGBoost"],
        skills_to_learn=[], learning_effort="low",
    )
]


def test_returns_string():
    agent = ReportWriterAgent(make_mock_client(SAMPLE_REPORT))
    result = agent.run(SCORED, critique_context="")
    assert isinstance(result, str)


def test_output_nonempty():
    agent = ReportWriterAgent(make_mock_client(SAMPLE_REPORT))
    result = agent.run(SCORED, critique_context="")
    assert len(result) > 100


def test_incorporates_critique_in_prompt():
    agent = ReportWriterAgent(make_mock_client(SAMPLE_REPORT))
    agent.run(SCORED, critique_context="TAM math needs source citations")
    call_args = agent.client.messages.create.call_args
    prompt_text = str(call_args)
    assert "TAM math needs source citations" in prompt_text
