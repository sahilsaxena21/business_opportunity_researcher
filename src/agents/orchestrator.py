import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tavily import TavilyClient

from src import config
from src.agents.consulting_trend import ConsultingTrendAgent
from src.agents.web_signal import WebSignalAgent
from src.agents.market_sizing import MarketSizingAgent
from src.agents.gap_analyst import GapAnalystAgent
from src.agents.skill_fit_scorer import SkillFitScorerAgent
from src.agents.report_writer import ReportWriterAgent
from src.agents.critic import CriticAgent
from src.models.schemas import (
    TrendCandidate, SignalCandidate, OpportunityCandidate,
    ScoredOpportunity, CritiqueResult, FinalReport, OpportunityBrief,
)
from src.search import make_search_fn


def merge_candidates(
    trends: List[TrendCandidate],
    signals: List[SignalCandidate],
) -> List[OpportunityCandidate]:
    """Combine trends and signals, deduplicating by title word overlap."""
    candidates: List[OpportunityCandidate] = []
    seen: set = set()

    for trend in trends:
        key = _normalize(trend.title)
        if key not in seen:
            seen.add(key)
            candidates.append(OpportunityCandidate(
                title=trend.title,
                description=trend.description,
                industry=trend.industry,
                sources=[trend.source],
            ))

    for signal in signals:
        key = _normalize(signal.title)
        overlapping = next(
            (c for c in candidates if _overlaps(_normalize(c.title), key)), None
        )
        if overlapping:
            if signal.source not in overlapping.sources:
                overlapping.sources.append(signal.source)
        else:
            seen.add(key)
            candidates.append(OpportunityCandidate(
                title=signal.title,
                description=signal.description,
                industry="unknown",
                sources=[signal.source],
            ))

    return candidates


def _normalize(title: str) -> str:
    return title.lower().replace("-", " ").replace("_", " ")


def _overlaps(a: str, b: str) -> bool:
    stopwords = {"the", "a", "an", "in", "of", "for", "and", "or", "with", "ai", "ml", "surge", "tools"}
    words_a = set(a.split()) - stopwords
    words_b = set(b.split()) - stopwords
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) >= 2


class Orchestrator:
    def __init__(self, tavily_client: "TavilyClient"):
        self.search_fn = make_search_fn(tavily_client)
        self.consulting_agent = ConsultingTrendAgent()
        self.web_signal_agent = WebSignalAgent()
        self.market_sizing_agent = MarketSizingAgent()
        self.gap_analyst_agent = GapAnalystAgent()
        self.skill_fit_scorer_agent = SkillFitScorerAgent()
        self.report_writer_agent = ReportWriterAgent()
        self.critic_agent = CriticAgent()

    def run(self, profile: dict) -> FinalReport:
        profile_summary = profile.get("raw", "")

        print("[1/6] Running research agents in parallel...")
        trends, signals = self._research_parallel(profile_summary)
        print(f"      {len(trends)} trends, {len(signals)} signals found")

        if not trends and not signals:
            return FinalReport(
                generated_at=datetime.now(),
                total_opportunities_found=0,
                ranked_opportunities=[],
                critique_rounds=[],
                report_markdown="# No opportunities found\n\nTry broadening search queries.",
            )

        print("[2/6] Merging candidates...")
        candidates = merge_candidates(trends, signals)
        print(f"      {len(candidates)} unique opportunities")

        print("[3/6] Sizing markets...")
        sized = self.market_sizing_agent.run(candidates)

        print("[4/6] Analyzing gaps...")
        gap_analyzed = self.gap_analyst_agent.run(sized)

        print("[5/6] Scoring skill fit...")
        scored = self.skill_fit_scorer_agent.run(gap_analyzed, profile)

        print("[6/6] Refinement loop (3 rounds)...")
        report_markdown, critique_rounds = self._refinement_loop(scored)

        ranked = sorted(scored, key=lambda x: x.reachable_fit_score, reverse=True)
        report = FinalReport(
            generated_at=datetime.now(),
            total_opportunities_found=len(scored),
            ranked_opportunities=_to_briefs(ranked),
            critique_rounds=critique_rounds,
            report_markdown=report_markdown,
        )
        self._save(report)
        return report

    def _research_parallel(self, profile_summary: str) -> Tuple[List[TrendCandidate], List[SignalCandidate]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            trends_f = pool.submit(self.consulting_agent.run, profile_summary, self.search_fn)
            signals_f = pool.submit(self.web_signal_agent.run, profile_summary, self.search_fn)
            return trends_f.result(), signals_f.result()

    def _refinement_loop(self, scored: List[ScoredOpportunity]) -> Tuple[str, List[CritiqueResult]]:
        critique_context = ""
        rounds: List[CritiqueResult] = []
        report_markdown = ""

        for i in range(1, config.REFINEMENT_ROUNDS + 1):
            print(f"      Round {i}/{config.REFINEMENT_ROUNDS}...")
            report_markdown = self.report_writer_agent.run(scored, critique_context)
            critique = self.critic_agent.run(report_markdown, round_number=i)
            rounds.append(critique)
            critique_context = (
                "\n".join(
                    f"- [{c.opportunity_title}] {c.issue}: {c.suggested_improvement}"
                    for c in critique.critiques
                )
                if critique.has_changes else ""
            )

        return report_markdown, rounds

    def _save(self, report: FinalReport) -> None:
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        ts = report.generated_at.strftime("%Y-%m-%d_%H-%M")
        path = Path(config.OUTPUT_DIR) / f"report_{ts}.md"
        path.write_text(report.report_markdown, encoding="utf-8")
        print(f"\nReport saved → {path}")


def _to_briefs(scored: List[ScoredOpportunity]) -> List[OpportunityBrief]:
    briefs = []
    for o in scored:
        title_lower = o.title.lower()
        industry_lower = o.industry.lower()

        if o.industry == "unknown" or "marketplace" in title_lower:
            leverage_model_type = "marketplace"
            recommended_business_model = f"Two-sided marketplace connecting {o.industry} buyers and sellers"
        elif "platform" in industry_lower or "platform" in title_lower:
            leverage_model_type = "platform"
            recommended_business_model = f"Platform play in {o.industry} with API/SDK monetization"
        else:
            leverage_model_type = "SaaS"
            recommended_business_model = f"B2B SaaS targeting {o.industry} companies"

        briefs.append(OpportunityBrief(
            **o.model_dump(),
            recommended_business_model=recommended_business_model,
            leverage_model_type=leverage_model_type,
            confidence_score=o.reachable_fit_score,
            confidence_rationale=o.fit_rationale,
        ))
    return briefs
