from pydantic import BaseModel
from datetime import datetime
from typing import List


class TrendCandidate(BaseModel):
    title: str
    description: str
    industry: str
    source: str


class SignalCandidate(BaseModel):
    title: str
    description: str
    signal_type: str   # "vc_investment", "job_spike", "community_buzz", "product_launch"
    source: str
    strength: str      # "strong", "moderate", "weak"


class OpportunityCandidate(BaseModel):
    title: str
    description: str
    industry: str
    sources: List[str]


class SizedOpportunity(OpportunityCandidate):
    tam_estimate: str  # e.g. "$2.4B"
    tam_math: str      # e.g. "48,000 dealers × $50,000/yr = $2.4B"
    tam_passes: bool   # True if TAM > $1B


class GapAnalyzedOpportunity(SizedOpportunity):
    gap_summary: str
    existing_solutions: List[str]
    key_gap: str


class ScoredOpportunity(GapAnalyzedOpportunity):
    current_fit_score: float     # 0.0 to 10.0 — based on existing skills only
    reachable_fit_score: float   # 0.0 to 10.0 — score if learnable skills are acquired
    fit_rationale: str           # explains both current fit and what can be learned
    relevant_skills: List[str]   # skills already possessed that apply
    skills_to_learn: List[str]   # skills needed but not yet held (learnable)
    learning_effort: str         # "low" (weeks–months) | "medium" (months–year) | "high" (years+)


class OpportunityBrief(ScoredOpportunity):
    recommended_business_model: str
    leverage_model_type: str   # "SaaS", "marketplace", "platform"
    confidence_score: float    # 0.0 to 10.0, final Critic rating
    confidence_rationale: str


class CritiqueItem(BaseModel):
    opportunity_title: str
    issue: str
    suggested_improvement: str


class CritiqueResult(BaseModel):
    round_number: int
    has_changes: bool
    critiques: List[CritiqueItem]
    overall_quality: str       # "poor", "fair", "good", "excellent"


class FinalReport(BaseModel):
    generated_at: datetime
    total_opportunities_found: int
    ranked_opportunities: List[OpportunityBrief]
    critique_rounds: List[CritiqueResult]
    report_markdown: str
