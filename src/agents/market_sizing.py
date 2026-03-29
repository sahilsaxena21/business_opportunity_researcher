import json
import re
from typing import List

from src.agents.base import BaseAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity

SYSTEM_PROMPT = """You are a market sizing analyst. For each opportunity, calculate the Total Addressable Market (TAM).

Formula: TAM = number of potential customers × annual price per customer
The opportunity passes if TAM > $1 billion. Show your math explicitly.

Return ONLY a valid JSON array extending each input with tam_estimate, tam_math, and tam_passes.
Preserve all original fields exactly. JSON array only — no markdown."""


class MarketSizingAgent(BaseAgent):
    def run(self, candidates: List[OpportunityCandidate]) -> List[SizedOpportunity]:
        if not candidates:
            return []

        user_prompt = f"""Calculate TAM for each opportunity.

For each one: estimate number of potential customers, realistic annual price, multiply to get TAM.
Mark tam_passes as true if TAM > $1B.

Opportunities:
{json.dumps([c.model_dump() for c in candidates], indent=2)}

Return full JSON array with tam_estimate (e.g. "$2.4B"), tam_math, and tam_passes added."""

        raw = self.call(SYSTEM_PROMPT, user_prompt)
        return _parse(raw)


def _parse(raw: str) -> List[SizedOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [SizedOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
