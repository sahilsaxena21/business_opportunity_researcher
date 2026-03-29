import json
import re
from typing import Callable, List

from src.agents.base import BaseAgent
from src.models.schemas import SignalCandidate

SYSTEM_PROMPT = """You are a market intelligence analyst identifying early signals of emerging business opportunities.

Signal types:
- vc_investment: recent VC funding rounds in a space
- job_spike: unusual growth in job postings for a domain
- community_buzz: high engagement on Reddit, HackerNews, Twitter/X
- product_launch: new products on Product Hunt indicating demand

Return ONLY a valid JSON array:
[{"title": "...", "description": "...", "signal_type": "...", "source": "...", "strength": "strong|moderate|weak"}]

JSON array only — no markdown, no explanation."""


class WebSignalAgent(BaseAgent):
    def run(self, profile_summary: str, search_fn: Callable[[str], str]) -> List[SignalCandidate]:
        user_prompt = f"""Search the web for emerging business opportunity signals.

Founder profile context:
{profile_summary}

Search for: recent VC investments in AI/ML, fraud tech, personalization, supply chain,
oil & gas tech. Also search Reddit, HackerNews, Product Hunt for trending problems.
Look for job posting spikes in relevant domains. Find at least 6 distinct signals.

Return JSON array only."""

        raw = self.call_with_search(SYSTEM_PROMPT, user_prompt, search_fn)
        return _parse(raw)


def _parse(raw: str) -> List[SignalCandidate]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [SignalCandidate(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
