import json
import re
from typing import List

from src.agents.base import BaseAgent
from src.models.schemas import GapAnalyzedOpportunity, ScoredOpportunity

SYSTEM_PROMPT = """You are a founder-opportunity fit analyst. Score each opportunity on TWO dimensions:

1. current_fit_score (0-10): Based only on skills the founder already has.
2. reachable_fit_score (0-10): Score if the founder learns the missing skills. An opportunity should NOT be penalized just because a skill gap exists — evaluate whether the gap is learnable and how hard.

Learning effort levels:
- "low": weeks to a few months (e.g. content marketing, basic SEO, no-code tools, a new industry's terminology)
- "medium": several months to a year (e.g. a new technical domain, sales motion, regulatory landscape)
- "high": multiple years (e.g. becoming a licensed professional, deep scientific expertise)

IMPORTANT: Do not restrict opportunities to only the founder's current domain. A healthcare opportunity with a "low" learning gap for a missing skill should still score well on reachable_fit_score.

Return ONLY a valid JSON array extending each input with:
- current_fit_score (float 0-10)
- reachable_fit_score (float 0-10)
- fit_rationale (string — explain both current fit AND what needs to be learned)
- relevant_skills (array of strings — skills already held that apply)
- skills_to_learn (array of strings — skills needed but not yet held)
- learning_effort ("low" | "medium" | "high")

Preserve all original fields exactly. JSON array only — no markdown."""


class SkillFitScorerAgent(BaseAgent):
    def run(self, candidates: List[GapAnalyzedOpportunity], profile: dict) -> List[ScoredOpportunity]:
        if not candidates:
            return []

        user_prompt = f"""Score the fit between this founder and each opportunity on both current and reachable dimensions.

Founder Profile:
{profile.get('raw', '')}

Existing skills to match against:
- Agentic AI pipelines (production LLM multi-step reasoning)
- Fraud detection (XGBoost, behavioral ML, clickstream data)
- Personalization engines (behavioral propensity models, real-time ML serving)
- MLOps (GCP, AWS, Docker, Airflow, end-to-end productionization)
- Supply chain / demand forecasting
- IoT anomaly detection (real-time systems)
- Oil & gas / EPCM (mechanical design, storage tanks, piping, stress analysis, Suncor/IOL clients)
- Procurement (RFQs, bid evaluations, Division 15 specs)
- Project delivery at scale ($40M+ in engineering projects)
- Cross-functional stakeholder management

For each opportunity, also identify any missing skills that are learnable (e.g. content marketing, a new industry domain, sales) and estimate learning effort. Do not penalize opportunities just because a skill gap exists.

Opportunities to score:
{json.dumps([c.model_dump() for c in candidates], indent=2)}

Return full JSON array with all six new fields added."""

        raw = self.call(SYSTEM_PROMPT, user_prompt)
        return _parse(raw)


def _parse(raw: str) -> List[ScoredOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [ScoredOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
