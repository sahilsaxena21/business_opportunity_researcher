import json
import re
from typing import List

from src.agents.base import BaseAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity

SYSTEM_PROMPT = """You are a competitive intelligence analyst. For each opportunity, identify:
1. 2-5 real existing solutions (company/product names)
2. The key gap — what critical need do existing solutions fail to address?
3. A gap summary — why this gap exists and why it matters (2-3 sentences)

Return ONLY a valid JSON array extending each input with gap_summary, existing_solutions (array), and key_gap.
Preserve all original fields exactly. JSON array only — no markdown."""


class GapAnalystAgent(BaseAgent):
    def run(self, candidates: List[SizedOpportunity]) -> List[GapAnalyzedOpportunity]:
        if not candidates:
            return []

        user_prompt = f"""Analyze the competitive landscape for each opportunity.

Name real existing solutions, identify the critical gap, and summarize why it exists.

Opportunities:
{json.dumps([c.model_dump() for c in candidates], indent=2)}

Return full JSON array with gap_summary, existing_solutions, and key_gap added."""

        raw = self.call(SYSTEM_PROMPT, user_prompt)
        return _parse(raw)


def _parse(raw: str) -> List[GapAnalyzedOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [GapAnalyzedOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
