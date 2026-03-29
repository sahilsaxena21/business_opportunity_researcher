import json
import re
from typing import Callable, List

from src.agents.base import BaseAgent
from src.models.schemas import TrendCandidate

SYSTEM_PROMPT = """You are a market research specialist extracting emerging business opportunities from consulting firm reports.

Your job:
1. Search for recent (2024-2026) reports from McKinsey, BCG, Bain, Deloitte, Gartner
2. Extract specific market trends representing new or underserved business opportunities
3. Focus on: rapid growth markets, technology disruption, regulatory-driven change

Return ONLY a valid JSON array:
[{"title": "...", "description": "...", "industry": "...", "source": "..."}]

JSON array only — no markdown, no explanation."""


class ConsultingTrendAgent(BaseAgent):
    def run(self, profile_summary: str, search_fn: Callable[[str], str]) -> List[TrendCandidate]:
        user_prompt = f"""Search consulting firm reports for emerging market trends.

Founder profile (use to prioritize relevant industries):
{profile_summary}

Search McKinsey, BCG, Bain, Deloitte for reports on: AI/ML enterprise, fintech fraud,
personalization technology, supply chain AI, oil & gas digital transformation, IoT analytics.
Find at least 8 distinct trends. Return JSON array only."""

        raw = self.call_with_search(SYSTEM_PROMPT, user_prompt, search_fn)
        return _parse(raw)


def _parse(raw: str) -> List[TrendCandidate]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [TrendCandidate(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
