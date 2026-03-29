import json
from typing import List

from src.agents.base import BaseAgent
from src.models.schemas import ScoredOpportunity

SYSTEM_PROMPT = """You are a business intelligence report writer. Produce a clear, structured markdown report.

The report must have two parts:
1. A ranked shortlist table (all opportunities, sorted by reachable_fit_score descending) with columns:
   Rank | Opportunity | Industry | TAM | Current Fit (0-10) | Reachable Fit (0-10) | Learning Effort | Leverage Model
   Flag opportunities where current_fit_score < 6 but reachable_fit_score >= 7 with a "🎯 Reach" label.
2. Deep-dive briefs for the top 5 opportunities (or all if fewer than 5), each containing:
   - The Trend (what market shift is happening, cite source)
   - The Gap (what existing solutions fail to do — name them specifically)
   - Market Size (TAM math: N customers × $price/yr = $TAM)
   - Your Edge (specific founder skills/experiences that apply — cite real roles and projects)
   - Skills to Learn (if any — what needs to be acquired and estimated time)
   - Recommended Business Model (recurring revenue, 70%+ margins, tech-scales, own product)
   - Confidence Score (0-10 with 1-sentence rationale)

Write direct, specific language. No vague statements that could apply to any founder."""


class ReportWriterAgent(BaseAgent):
    def run(self, scored_opportunities: List[ScoredOpportunity], critique_context: str) -> str:
        """Produce a markdown report, incorporating any critique from a prior round."""
        ranked = sorted(scored_opportunities, key=lambda x: x.reachable_fit_score, reverse=True)
        candidates_json = json.dumps([o.model_dump() for o in ranked], indent=2)

        critique_section = (
            f"\n\nAddress these issues from the previous round:\n{critique_context}"
            if critique_context else ""
        )

        user_prompt = f"""Write a business opportunity report from this data.{critique_section}

Scored opportunities (sorted by fit score):
{candidates_json}

Produce the ranked shortlist table and top-5 deep-dive briefs."""

        return self.call(SYSTEM_PROMPT, user_prompt)
