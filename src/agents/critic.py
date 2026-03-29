import json
import re

from src.agents.base import BaseAgent
from src.models.schemas import CritiqueResult, CritiqueItem

SYSTEM_PROMPT = """You are a rigorous business research critic. Review a business opportunity report for weaknesses.

Check for:
1. Unsupported TAM claims — is the customer count or price estimate cited or justified?
2. Shallow gap analysis — are existing solutions named specifically, or just vaguely described?
3. Weak fit scoring — is the rationale specific and tied to real founder experiences?
4. Missing business model logic — does the model meet high-leverage criteria (recurring, 70%+ margins, tech-scales, own product)?
5. Vague language — any claims that could apply to any founder or market?

Return ONLY a valid JSON object:
{
  "round_number": <int>,
  "has_changes": <bool>,
  "critiques": [{"opportunity_title": "...", "issue": "...", "suggested_improvement": "..."}],
  "overall_quality": "poor|fair|good|excellent"
}

If no significant issues exist, set has_changes to false and critiques to []. JSON only — no markdown."""


class CriticAgent(BaseAgent):
    def run(self, report_markdown: str, round_number: int) -> CritiqueResult:
        user_prompt = f"""Review this business opportunity report (round {round_number} of 3).

{report_markdown}

Identify specific weaknesses. Name the exact opportunity and the exact issue. Return the JSON object."""

        raw = self.call(SYSTEM_PROMPT, user_prompt)
        return _parse(raw, round_number)


def _parse(raw: str, round_number: int) -> CritiqueResult:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return CritiqueResult(round_number=round_number, has_changes=False, critiques=[], overall_quality="good")
    try:
        data = json.loads(match.group())
        return CritiqueResult(
            round_number=round_number,
            has_changes=data.get("has_changes", False),
            critiques=[CritiqueItem(**c) for c in data.get("critiques", [])],
            overall_quality=data.get("overall_quality", "good"),
        )
    except (json.JSONDecodeError, TypeError):
        return CritiqueResult(round_number=round_number, has_changes=False, critiques=[], overall_quality="good")
