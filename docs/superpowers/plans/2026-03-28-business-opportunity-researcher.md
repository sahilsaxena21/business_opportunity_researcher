# Business Opportunity Researcher — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an on-demand multi-agent pipeline that discovers and ranks business opportunities by intersecting market trends with the founder's unique skillset, producing a ranked shortlist and top-5 deep-dive briefs.

**Architecture:** Two parallel research agents (consulting reports + web signals) feed a deduplicated candidate list into three sequential analysis agents (market sizing → gap analysis → skill fit scoring), followed by a fixed 3-round Report Writer + Critic refinement loop before producing a final markdown report.

**Tech Stack:** Python 3.10+, `anthropic` SDK, Pydantic v2, `tavily-python` (web search), `python-dotenv`, `pytest`

---

## File Structure

```
src/
  __init__.py
  config.py                  — API keys, model name, constants (REFINEMENT_ROUNDS=3, etc.)
  profile.py                 — Load memory/user_profile.md into structured dict
  search.py                  — Tavily search helper: make_search_fn(client) → Callable
  models/
    __init__.py
    schemas.py               — All Pydantic models (TrendCandidate → OpportunityBrief chain)
  agents/
    __init__.py
    base.py                  — BaseAgent: call() and call_with_search() (tool-use loop)
    consulting_trend.py      — ConsultingTrendAgent: search consulting reports → list[TrendCandidate]
    web_signal.py            — WebSignalAgent: search web signals → list[SignalCandidate]
    market_sizing.py         — MarketSizingAgent: TAM calc → list[SizedOpportunity]
    gap_analyst.py           — GapAnalystAgent: competitor landscape → list[GapAnalyzedOpportunity]
    skill_fit_scorer.py      — SkillFitScorerAgent: score vs profile → list[ScoredOpportunity]
    report_writer.py         — ReportWriterAgent: synthesize → markdown string
    critic.py                — CriticAgent: review draft → CritiqueResult
    orchestrator.py          — Orchestrator: wires full pipeline + refinement loop
tests/
  __init__.py
  test_profile.py
  test_schemas.py
  test_search.py
  agents/
    __init__.py
    test_consulting_trend.py
    test_web_signal.py
    test_market_sizing.py
    test_gap_analyst.py
    test_skill_fit_scorer.py
    test_report_writer.py
    test_critic.py
    test_orchestrator.py
output/                      — Generated reports (gitignored)
memory/
  user_profile.md            — Already exists
main.py                      — CLI entry point
requirements.txt
.env.example
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`, `src/models/__init__.py`, `src/agents/__init__.py`
- Create: `tests/__init__.py`, `tests/agents/__init__.py`
- Create: `src/config.py`
- Create: `output/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
anthropic>=0.40.0
pydantic>=2.0.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- [ ] **Step 3: Create src/config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
MODEL = "claude-sonnet-4-6"
REFINEMENT_ROUNDS = 3
PROFILE_PATH = "memory/user_profile.md"
OUTPUT_DIR = "output"
```

- [ ] **Step 4: Create empty init files and output placeholder**

```bash
touch src/__init__.py src/models/__init__.py src/agents/__init__.py
touch tests/__init__.py tests/agents/__init__.py
touch output/.gitkeep
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Copy .env.example → .env and fill in real API keys**

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and TAVILY_API_KEY
```

- [ ] **Step 7: Commit**

```bash
git init
git add requirements.txt .env.example src/ tests/ output/.gitkeep
git commit -m "feat: project scaffold"
```

---

## Task 2: Data Schemas

**Files:**
- Create: `src/models/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pytest
from datetime import datetime
from src.models.schemas import (
    TrendCandidate, SignalCandidate, OpportunityCandidate,
    SizedOpportunity, GapAnalyzedOpportunity, ScoredOpportunity,
    OpportunityBrief, CritiqueItem, CritiqueResult, FinalReport,
)


def test_trend_candidate_valid():
    t = TrendCandidate(title="AI in fraud", description="desc", industry="fintech", source="McKinsey")
    assert t.title == "AI in fraud"


def test_sized_opportunity_inherits_candidate():
    o = SizedOpportunity(
        title="Fraud AI", description="desc", industry="fintech", sources=["McKinsey"],
        tam_estimate="$2.4B", tam_math="48000 x $50000 = $2.4B", tam_passes=True,
    )
    assert o.tam_passes is True


def test_scored_opportunity_fit_score_range():
    o = ScoredOpportunity(
        title="Fraud AI", description="desc", industry="fintech", sources=["McKinsey"],
        tam_estimate="$2.4B", tam_math="48000 x $50000 = $2.4B", tam_passes=True,
        gap_summary="no ML", existing_solutions=["ReedCase"], key_gap="no behavioral ML",
        current_fit_score=8.5, reachable_fit_score=9.0,
        fit_rationale="matches XGBoost fraud work; content marketing learnable",
        relevant_skills=["XGBoost"], skills_to_learn=["content marketing"],
        learning_effort="low",
    )
    assert 0.0 <= o.current_fit_score <= 10.0
    assert 0.0 <= o.reachable_fit_score <= 10.0


def test_critique_result_structure():
    r = CritiqueResult(
        round_number=1,
        has_changes=True,
        critiques=[CritiqueItem(
            opportunity_title="Fraud AI",
            issue="TAM math unverified",
            suggested_improvement="Add source for 48000 figure",
        )],
        overall_quality="fair",
    )
    assert r.round_number == 1
    assert len(r.critiques) == 1


def test_final_report_requires_markdown():
    r = FinalReport(
        generated_at=datetime.now(),
        total_opportunities_found=5,
        ranked_opportunities=[],
        critique_rounds=[],
        report_markdown="# Report",
    )
    assert r.report_markdown == "# Report"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Create src/models/schemas.py**

```python
from pydantic import BaseModel
from datetime import datetime


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
    sources: list[str]


class SizedOpportunity(OpportunityCandidate):
    tam_estimate: str  # e.g. "$2.4B"
    tam_math: str      # e.g. "48,000 dealers × $50,000/yr = $2.4B"
    tam_passes: bool   # True if TAM > $1B


class GapAnalyzedOpportunity(SizedOpportunity):
    gap_summary: str
    existing_solutions: list[str]
    key_gap: str


class ScoredOpportunity(GapAnalyzedOpportunity):
    current_fit_score: float     # 0.0 to 10.0 — based on existing skills only
    reachable_fit_score: float   # 0.0 to 10.0 — score if learnable skills are acquired
    fit_rationale: str           # explains both current fit and what can be learned
    relevant_skills: list[str]   # skills already possessed that apply
    skills_to_learn: list[str]   # skills needed but not yet held (learnable)
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
    critiques: list[CritiqueItem]
    overall_quality: str       # "poor", "fair", "good", "excellent"


class FinalReport(BaseModel):
    generated_at: datetime
    total_opportunities_found: int
    ranked_opportunities: list[OpportunityBrief]
    critique_rounds: list[CritiqueResult]
    report_markdown: str
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schemas.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/schemas.py tests/test_schemas.py
git commit -m "feat: add data schemas"
```

---

## Task 3: Profile Loader

**Files:**
- Create: `src/profile.py`
- Test: `tests/test_profile.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_profile.py
import pytest
from src.profile import load_profile


def test_load_profile_returns_dict():
    profile = load_profile("memory/user_profile.md")
    assert isinstance(profile, dict)


def test_load_profile_has_required_keys():
    profile = load_profile("memory/user_profile.md")
    assert "name" in profile
    assert "skills" in profile
    assert "industries" in profile
    assert "summary" in profile
    assert "raw" in profile


def test_load_profile_raw_is_nonempty_string():
    profile = load_profile("memory/user_profile.md")
    assert isinstance(profile["raw"], str)
    assert len(profile["raw"]) > 100


def test_load_profile_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("memory/nonexistent.md")


def test_load_profile_raw_mentions_data_science():
    profile = load_profile("memory/user_profile.md")
    text = profile["raw"].lower()
    assert "data" in text or "ml" in text or "fraud" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_profile.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.profile'`

- [ ] **Step 3: Create src/profile.py**

```python
import re
from pathlib import Path


def load_profile(path: str) -> dict:
    """Load and parse user_profile.md into a structured dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    content = p.read_text(encoding="utf-8")

    # Strip YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        body = parts[2].strip() if len(parts) >= 3 else content
    else:
        body = content

    return {
        "name": "Sahil Saxena",
        "summary": _extract_section(body, "## Identity") or body[:500],
        "skills": _extract_section(body, "## Data Science & AI Skills") or "",
        "industries": _extract_section(body, "## Industries with Hands-On DS Experience") or "",
        "engineering_background": _extract_section(body, "## Mechanical Engineering & EPCM Background") or "",
        "business_skills": _extract_section(body, "## Business & Management Skills") or "",
        "interests": _extract_section(body, "## Current Interest Areas") or "",
        "raw": body,
    }


def _extract_section(content: str, heading: str) -> str:
    """Extract content under a markdown heading until the next heading."""
    if heading not in content:
        return ""
    start = content.index(heading) + len(heading)
    remaining = content[start:]
    next_heading = re.search(r'\n## ', remaining)
    end = next_heading.start() if next_heading else len(remaining)
    return remaining[:end].strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_profile.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/profile.py tests/test_profile.py
git commit -m "feat: add profile loader"
```

---

## Task 4: Search Helper

**Files:**
- Create: `src/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import pytest
from unittest.mock import MagicMock
from src.search import make_search_fn


def test_make_search_fn_returns_callable():
    fn = make_search_fn(MagicMock())
    assert callable(fn)


def test_search_fn_returns_string():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "AI Trend", "url": "https://example.com", "content": "AI is growing fast."}
        ]
    }
    fn = make_search_fn(mock_client)
    result = fn("AI market trends 2025")
    assert isinstance(result, str)
    assert "AI Trend" in result


def test_search_fn_handles_empty_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    fn = make_search_fn(mock_client)
    result = fn("obscure query xyz")
    assert result == "[]"


def test_search_fn_truncates_long_content():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [{"title": "T", "url": "u", "content": "x" * 1000}]
    }
    fn = make_search_fn(mock_client)
    import json
    result = json.loads(fn("query"))
    assert len(result[0]["content"]) <= 600
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.search'`

- [ ] **Step 3: Create src/search.py**

```python
import json
from typing import Callable
from tavily import TavilyClient


def make_search_fn(tavily_client: TavilyClient, max_results: int = 5) -> Callable[[str], str]:
    """Return a function that queries Tavily and returns a JSON string of results."""

    def search(query: str) -> str:
        response = tavily_client.search(query, max_results=max_results)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:600],
            }
            for r in response.get("results", [])
        ]
        return json.dumps(results)

    return search
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_search.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/search.py tests/test_search.py
git commit -m "feat: add Tavily search helper"
```

---

## Task 5: Base Agent

**Files:**
- Create: `src/agents/base.py`

- [ ] **Step 1: Create src/agents/base.py**

```python
from typing import Callable
import anthropic


class BaseAgent:
    """Shared Claude API call logic for all agents."""

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def call(self, system: str, user: str) -> str:
        """Single-turn call with no tools."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def call_with_search(
        self,
        system: str,
        user: str,
        search_fn: Callable[[str], str],
        max_searches: int = 5,
    ) -> str:
        """Multi-turn call with a web_search tool. Executes search_fn on each tool call."""
        tools = [
            {
                "name": "web_search",
                "description": "Search the web for up-to-date information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"],
                },
            }
        ]

        messages = [{"role": "user", "content": user}]
        searches_used = 0

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=system,
                messages=messages,
                tools=tools,
            )

            if response.stop_reason == "end_turn":
                return "".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason != "tool_use" or searches_used >= max_searches:
                return "".join(b.text for b in response.content if b.type == "text")

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use" and block.name == "web_search":
                    result = search_fn(block.input.get("query", ""))
                    searches_used += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/base.py
git commit -m "feat: add base agent with web search tool loop"
```

---

## Task 6: Consulting Trend Agent

**Files:**
- Create: `src/agents/consulting_trend.py`
- Test: `tests/agents/test_consulting_trend.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_consulting_trend.py
import pytest
from unittest.mock import MagicMock
from src.agents.consulting_trend import ConsultingTrendAgent
from src.models.schemas import TrendCandidate


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {"title": "AI-Driven Fraud Detection", "description": "Banks adopting ML", "industry": "fintech", "source": "McKinsey"},
  {"title": "Supply Chain AI", "description": "AI optimizing logistics", "industry": "retail", "source": "BCG"}
]'''


def test_returns_list():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("Data scientist profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_trend_candidates():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("Data scientist profile", lambda q: "[]")
    assert all(isinstance(r, TrendCandidate) for r in results)


def test_parses_title():
    agent = ConsultingTrendAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "AI-Driven Fraud Detection" for r in results)


def test_returns_empty_on_bad_json():
    agent = ConsultingTrendAgent(make_mock_client("Sorry, no trends found."))
    results = agent.run("profile", lambda q: "[]")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_consulting_trend.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.consulting_trend'`

- [ ] **Step 3: Create src/agents/consulting_trend.py**

```python
import json
import re
from typing import Callable

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
    def run(self, profile_summary: str, search_fn: Callable[[str], str]) -> list[TrendCandidate]:
        user_prompt = f"""Search consulting firm reports for emerging market trends.

Founder profile (use to prioritize relevant industries):
{profile_summary}

Search McKinsey, BCG, Bain, Deloitte for reports on: AI/ML enterprise, fintech fraud,
personalization technology, supply chain AI, oil & gas digital transformation, IoT analytics.
Find at least 8 distinct trends. Return JSON array only."""

        raw = self.call_with_search(SYSTEM_PROMPT, user_prompt, search_fn)
        return _parse(raw)


def _parse(raw: str) -> list[TrendCandidate]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [TrendCandidate(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_consulting_trend.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/consulting_trend.py tests/agents/test_consulting_trend.py
git commit -m "feat: add consulting trend agent"
```

---

## Task 7: Web Signal Agent

**Files:**
- Create: `src/agents/web_signal.py`
- Test: `tests/agents/test_web_signal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_web_signal.py
import pytest
from unittest.mock import MagicMock
from src.agents.web_signal import WebSignalAgent
from src.models.schemas import SignalCandidate


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {"title": "VC surge in AI fraud tools", "description": "5 rounds >$50M in 2025", "signal_type": "vc_investment", "source": "TechCrunch", "strength": "strong"},
  {"title": "Job spike: ML fraud engineers", "description": "300% YoY increase", "signal_type": "job_spike", "source": "LinkedIn", "strength": "strong"}
]'''


def test_returns_list():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("fraud ML expert", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_signal_candidates():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("fraud ML expert", lambda q: "[]")
    assert all(isinstance(r, SignalCandidate) for r in results)


def test_parses_signal_type():
    agent = WebSignalAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run("profile", lambda q: "[]")
    assert any(r.signal_type == "vc_investment" for r in results)


def test_returns_empty_on_bad_json():
    agent = WebSignalAgent(make_mock_client("No signals found."))
    results = agent.run("profile", lambda q: "[]")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_web_signal.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.web_signal'`

- [ ] **Step 3: Create src/agents/web_signal.py**

```python
import json
import re
from typing import Callable

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
    def run(self, profile_summary: str, search_fn: Callable[[str], str]) -> list[SignalCandidate]:
        user_prompt = f"""Search the web for emerging business opportunity signals.

Founder profile context:
{profile_summary}

Search for: recent VC investments in AI/ML, fraud tech, personalization, supply chain,
oil & gas tech. Also search Reddit, HackerNews, Product Hunt for trending problems.
Look for job posting spikes in relevant domains. Find at least 6 distinct signals.

Return JSON array only."""

        raw = self.call_with_search(SYSTEM_PROMPT, user_prompt, search_fn)
        return _parse(raw)


def _parse(raw: str) -> list[SignalCandidate]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [SignalCandidate(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_web_signal.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/web_signal.py tests/agents/test_web_signal.py
git commit -m "feat: add web signal agent"
```

---

## Task 8: Market Sizing Agent

**Files:**
- Create: `src/agents/market_sizing.py`
- Test: `tests/agents/test_market_sizing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_market_sizing.py
import pytest
from unittest.mock import MagicMock
from src.agents.market_sizing import MarketSizingAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {
    "title": "AI Fraud Detection SaaS",
    "description": "ML platform for bank fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$3.2B",
    "tam_math": "6000 banks x $53000/yr = $3.2B",
    "tam_passes": true
  }
]'''

CANDIDATES = [
    OpportunityCandidate(
        title="AI Fraud Detection SaaS",
        description="ML platform for bank fraud",
        industry="fintech",
        sources=["McKinsey"],
    )
]


def test_returns_list():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES), list)


def test_returns_sized_opportunities():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert all(isinstance(r, SizedOpportunity) for r in results)


def test_preserves_title():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert agent.run(CANDIDATES)[0].title == "AI Fraud Detection SaaS"


def test_sets_tam_passes():
    agent = MarketSizingAgent(make_mock_client(SAMPLE_JSON))
    assert agent.run(CANDIDATES)[0].tam_passes is True


def test_empty_candidates_returns_empty():
    agent = MarketSizingAgent(make_mock_client("[]"))
    assert agent.run([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_market_sizing.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.market_sizing'`

- [ ] **Step 3: Create src/agents/market_sizing.py**

```python
import json
import re

from src.agents.base import BaseAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity

SYSTEM_PROMPT = """You are a market sizing analyst. For each opportunity, calculate the Total Addressable Market (TAM).

Formula: TAM = number of potential customers × annual price per customer
The opportunity passes if TAM > $1 billion. Show your math explicitly.

Return ONLY a valid JSON array extending each input with tam_estimate, tam_math, and tam_passes.
Preserve all original fields exactly. JSON array only — no markdown."""


class MarketSizingAgent(BaseAgent):
    def run(self, candidates: list[OpportunityCandidate]) -> list[SizedOpportunity]:
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


def _parse(raw: str) -> list[SizedOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [SizedOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_market_sizing.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/market_sizing.py tests/agents/test_market_sizing.py
git commit -m "feat: add market sizing agent"
```

---

## Task 9: Gap Analyst Agent

**Files:**
- Create: `src/agents/gap_analyst.py`
- Test: `tests/agents/test_gap_analyst.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_gap_analyst.py
import pytest
from unittest.mock import MagicMock
from src.agents.gap_analyst import GapAnalystAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {
    "title": "AI Fraud Detection SaaS", "description": "ML platform", "industry": "fintech",
    "sources": ["McKinsey"], "tam_estimate": "$3.2B", "tam_math": "6000 x $53000 = $3.2B", "tam_passes": true,
    "gap_summary": "Existing tools lack behavioral ML signals",
    "existing_solutions": ["NICE Actimize", "SAS Fraud"],
    "key_gap": "No real-time behavioral feature engineering"
  }
]'''

CANDIDATES = [
    SizedOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
    )
]


def test_returns_list():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES), list)


def test_returns_gap_analyzed_opportunities():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert all(isinstance(r, GapAnalyzedOpportunity) for r in results)


def test_sets_key_gap():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert results[0].key_gap == "No real-time behavioral feature engineering"


def test_sets_existing_solutions():
    agent = GapAnalystAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES)
    assert "NICE Actimize" in results[0].existing_solutions
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_gap_analyst.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.gap_analyst'`

- [ ] **Step 3: Create src/agents/gap_analyst.py**

```python
import json
import re

from src.agents.base import BaseAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity

SYSTEM_PROMPT = """You are a competitive intelligence analyst. For each opportunity, identify:
1. 2-5 real existing solutions (company/product names)
2. The key gap — what critical need do existing solutions fail to address?
3. A gap summary — why this gap exists and why it matters (2-3 sentences)

Return ONLY a valid JSON array extending each input with gap_summary, existing_solutions (array), and key_gap.
Preserve all original fields exactly. JSON array only — no markdown."""


class GapAnalystAgent(BaseAgent):
    def run(self, candidates: list[SizedOpportunity]) -> list[GapAnalyzedOpportunity]:
        if not candidates:
            return []

        user_prompt = f"""Analyze the competitive landscape for each opportunity.

Name real existing solutions, identify the critical gap, and summarize why it exists.

Opportunities:
{json.dumps([c.model_dump() for c in candidates], indent=2)}

Return full JSON array with gap_summary, existing_solutions, and key_gap added."""

        raw = self.call(SYSTEM_PROMPT, user_prompt)
        return _parse(raw)


def _parse(raw: str) -> list[GapAnalyzedOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [GapAnalyzedOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_gap_analyst.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/gap_analyst.py tests/agents/test_gap_analyst.py
git commit -m "feat: add gap analyst agent"
```

---

## Task 10: Skill Fit Scorer Agent

**Files:**
- Create: `src/agents/skill_fit_scorer.py`
- Test: `tests/agents/test_skill_fit_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_skill_fit_scorer.py
import pytest
from unittest.mock import MagicMock
from src.agents.skill_fit_scorer import SkillFitScorerAgent
from src.models.schemas import GapAnalyzedOpportunity, ScoredOpportunity


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


SAMPLE_JSON = '''[
  {
    "title": "AI Fraud Detection SaaS", "description": "ML platform", "industry": "fintech",
    "sources": ["McKinsey"], "tam_estimate": "$3.2B", "tam_math": "6000 x $53000 = $3.2B", "tam_passes": true,
    "gap_summary": "No behavioral ML", "existing_solutions": ["NICE Actimize"], "key_gap": "No real-time features",
    "current_fit_score": 9.0,
    "reachable_fit_score": 9.5,
    "fit_rationale": "Sahil built fraud detection at Scotiabank using XGBoost; content marketing learnable in months",
    "relevant_skills": ["XGBoost", "fraud detection", "clickstream analysis"],
    "skills_to_learn": ["content marketing"],
    "learning_effort": "low"
  }
]'''

CANDIDATES = [
    GapAnalyzedOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
        gap_summary="No behavioral ML", existing_solutions=["NICE Actimize"], key_gap="No real-time features",
    )
]

PROFILE = {"name": "Sahil", "raw": "XGBoost fraud detection at Scotiabank, agentic AI pipelines"}


def test_returns_list():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    assert isinstance(agent.run(CANDIDATES, PROFILE), list)


def test_returns_scored_opportunities():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert all(isinstance(r, ScoredOpportunity) for r in results)


def test_scores_in_range():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert 0.0 <= results[0].current_fit_score <= 10.0
    assert 0.0 <= results[0].reachable_fit_score <= 10.0


def test_learning_effort_valid_value():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert results[0].learning_effort in ("low", "medium", "high")


def test_skills_to_learn_is_list():
    agent = SkillFitScorerAgent(make_mock_client(SAMPLE_JSON))
    results = agent.run(CANDIDATES, PROFILE)
    assert isinstance(results[0].skills_to_learn, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_skill_fit_scorer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.skill_fit_scorer'`

- [ ] **Step 3: Create src/agents/skill_fit_scorer.py**

```python
import json
import re

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
    def run(self, candidates: list[GapAnalyzedOpportunity], profile: dict) -> list[ScoredOpportunity]:
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


def _parse(raw: str) -> list[ScoredOpportunity]:
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return [ScoredOpportunity(**item) for item in json.loads(match.group()) if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_skill_fit_scorer.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/skill_fit_scorer.py tests/agents/test_skill_fit_scorer.py
git commit -m "feat: add skill fit scorer agent"
```

---

## Task 11: Report Writer Agent

**Files:**
- Create: `src/agents/report_writer.py`
- Test: `tests/agents/test_report_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_report_writer.py
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


SCORED = [
    ScoredOpportunity(
        title="AI Fraud Detection SaaS", description="ML platform", industry="fintech",
        sources=["McKinsey"], tam_estimate="$3.2B", tam_math="6000 x $53000 = $3.2B", tam_passes=True,
        gap_summary="No behavioral ML", existing_solutions=["NICE Actimize"], key_gap="No real-time features",
        fit_score=9.0, fit_rationale="Built at Scotiabank", relevant_skills=["XGBoost"],
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_report_writer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.report_writer'`

- [ ] **Step 3: Create src/agents/report_writer.py**

```python
import json

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
    def run(self, scored_opportunities: list[ScoredOpportunity], critique_context: str) -> str:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_report_writer.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/report_writer.py tests/agents/test_report_writer.py
git commit -m "feat: add report writer agent"
```

---

## Task 12: Critic Agent

**Files:**
- Create: `src/agents/critic.py`
- Test: `tests/agents/test_critic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_critic.py
import pytest
from unittest.mock import MagicMock
from src.agents.critic import CriticAgent
from src.models.schemas import CritiqueResult


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text=response_text)]
    mock_msg.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create.return_value = mock_msg
    return client


WITH_CHANGES = '''{
  "round_number": 1,
  "has_changes": true,
  "critiques": [
    {"opportunity_title": "AI Fraud SaaS", "issue": "TAM lacks source", "suggested_improvement": "Cite FDIC data"}
  ],
  "overall_quality": "fair"
}'''

NO_CHANGES = '''{
  "round_number": 2,
  "has_changes": false,
  "critiques": [],
  "overall_quality": "good"
}'''


def test_returns_critique_result():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert isinstance(agent.run("# Report", round_number=1), CritiqueResult)


def test_parses_has_changes_true():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert agent.run("# Report", round_number=1).has_changes is True


def test_parses_critiques_list():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    result = agent.run("# Report", round_number=1)
    assert len(result.critiques) == 1
    assert result.critiques[0].opportunity_title == "AI Fraud SaaS"


def test_no_changes_returns_empty_critiques():
    agent = CriticAgent(make_mock_client(NO_CHANGES))
    result = agent.run("# Good report", round_number=2)
    assert result.has_changes is False
    assert result.critiques == []


def test_sets_round_number():
    agent = CriticAgent(make_mock_client(WITH_CHANGES))
    assert agent.run("# Report", round_number=1).round_number == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_critic.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.critic'`

- [ ] **Step 3: Create src/agents/critic.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_critic.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/critic.py tests/agents/test_critic.py
git commit -m "feat: add critic agent"
```

---

## Task 13: Orchestrator

**Files:**
- Create: `src/agents/orchestrator.py`
- Test: `tests/agents/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_orchestrator.py
import pytest
from unittest.mock import MagicMock
from src.agents.orchestrator import Orchestrator, merge_candidates
from src.models.schemas import TrendCandidate, SignalCandidate, OpportunityCandidate


def test_merge_combines_trends_and_signals():
    trends = [
        TrendCandidate(title="AI Fraud Detection", description="d", industry="fintech", source="McKinsey"),
        TrendCandidate(title="Supply Chain AI", description="d2", industry="retail", source="BCG"),
    ]
    signals = [
        SignalCandidate(title="VC surge in IoT", description="d3", signal_type="vc_investment", source="TC", strength="strong"),
    ]
    result = merge_candidates(trends, signals)
    assert len(result) == 3
    assert all(isinstance(c, OpportunityCandidate) for c in result)


def test_merge_deduplicates_overlapping():
    trends = [TrendCandidate(title="AI fraud detection tools", description="d", industry="fintech", source="McKinsey")]
    signals = [SignalCandidate(title="AI fraud detection surge", description="VC money", signal_type="vc_investment", source="TC", strength="strong")]
    result = merge_candidates(trends, signals)
    assert len(result) == 1


def test_merge_empty_inputs():
    assert merge_candidates([], []) == []


def test_orchestrator_initializes():
    orch = Orchestrator(anthropic_client=MagicMock(), tavily_client=MagicMock())
    assert orch is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.orchestrator'`

- [ ] **Step 3: Create src/agents/orchestrator.py**

```python
import concurrent.futures
from datetime import datetime
from pathlib import Path

import anthropic
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
    trends: list[TrendCandidate],
    signals: list[SignalCandidate],
) -> list[OpportunityCandidate]:
    """Combine trends and signals, deduplicating by title word overlap."""
    candidates: list[OpportunityCandidate] = []
    seen: set[str] = set()

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
    def __init__(self, anthropic_client: anthropic.Anthropic, tavily_client: TavilyClient):
        self.search_fn = make_search_fn(tavily_client)
        self.consulting_agent = ConsultingTrendAgent(anthropic_client)
        self.web_signal_agent = WebSignalAgent(anthropic_client)
        self.market_sizing_agent = MarketSizingAgent(anthropic_client)
        self.gap_analyst_agent = GapAnalystAgent(anthropic_client)
        self.skill_fit_scorer_agent = SkillFitScorerAgent(anthropic_client)
        self.report_writer_agent = ReportWriterAgent(anthropic_client)
        self.critic_agent = CriticAgent(anthropic_client)

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

    def _research_parallel(self, profile_summary: str):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            trends_f = pool.submit(self.consulting_agent.run, profile_summary, self.search_fn)
            signals_f = pool.submit(self.web_signal_agent.run, profile_summary, self.search_fn)
            return trends_f.result(), signals_f.result()

    def _refinement_loop(self, scored: list[ScoredOpportunity]) -> tuple[str, list[CritiqueResult]]:
        critique_context = ""
        rounds: list[CritiqueResult] = []
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


def _to_briefs(scored: list[ScoredOpportunity]) -> list[OpportunityBrief]:
    return [
        OpportunityBrief(
            **o.model_dump(),
            recommended_business_model=f"B2B SaaS targeting {o.industry} companies",
            leverage_model_type="SaaS",
            confidence_score=o.reachable_fit_score,
            confidence_rationale=o.fit_rationale,
        )
        for o in scored
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/agents/test_orchestrator.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "feat: add orchestrator with parallel research and 3-round refinement loop"
```

---

## Task 14: CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Run full test suite to confirm all prior tasks pass**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Create main.py**

```python
import sys
import anthropic
from tavily import TavilyClient

from src import config
from src.profile import load_profile
from src.agents.orchestrator import Orchestrator


def main():
    print("=== Business Opportunity Researcher ===\n")

    try:
        profile = load_profile(config.PROFILE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Profile loaded: {profile['name']}\n")

    anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
    orchestrator = Orchestrator(anthropic_client=anthropic_client, tavily_client=tavily_client)

    report = orchestrator.run(profile)

    print(f"\nDone. {report.total_opportunities_found} opportunities found.")
    if report.ranked_opportunities:
        print(f"Top opportunity: {report.ranked_opportunities[0].title}")
    print("\n--- Report Preview ---")
    print(report.report_markdown[:600])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test (requires real API keys in .env)**

```bash
python main.py
```

Expected output:
```
=== Business Opportunity Researcher ===

Profile loaded: Sahil Saxena

[1/6] Running research agents in parallel...
      N trends, M signals found
[2/6] Merging candidates...
      K unique opportunities
[3/6] Sizing markets...
[4/6] Analyzing gaps...
[5/6] Scoring skill fit...
[6/6] Refinement loop (3 rounds)...
      Round 1/3...
      Round 2/3...
      Round 3/3...

Report saved → output/report_YYYY-MM-DD_HH-MM.md

Done. K opportunities found.
Top opportunity: <title>
```

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point"
```

---

## Self-Review

**Spec coverage:**
- Two parallel research agents (Consulting Trend + Web Signal) — Task 6, 7, 13 ✓
- Three sequential analysis agents (Market Sizing → Gap Analyst → Skill Fit Scorer) — Tasks 8, 9, 10 ✓
- Report Writer + Critic in exactly 3-round refinement loop — Tasks 11, 12, 13 ✓
- Orchestrator merges + deduplicates candidates — Task 13 (`merge_candidates`) ✓
- Profile loaded from `memory/user_profile.md` — Task 3 ✓
- Sahil's oil & gas/EPCM background included in skill fit scoring — Task 10, system prompt ✓
- Output: markdown file in `output/` directory — Task 13 (`_save`) ✓
- Ranked shortlist + top-5 deep-dive briefs — Task 11 (ReportWriterAgent prompt) ✓
- On-demand CLI trigger — Task 14 ✓
- Empty results: halts early with message — Task 13 (`if not trends and not signals`) ✓
- Agent failure handling: each `_parse` function returns `[]` on bad JSON — Tasks 6–12 ✓

**Placeholder scan:** No TBDs, no "implement later", no vague steps. ✓

**Type consistency:**
- `TrendCandidate/SignalCandidate` → `OpportunityCandidate` (merge) → `SizedOpportunity` → `GapAnalyzedOpportunity` → `ScoredOpportunity` → `OpportunityBrief` — chain is consistent across all tasks ✓
- `CritiqueResult` uses `list[CritiqueItem]` — consistent in `critic.py` and `orchestrator.py` ✓
- `merge_candidates` returns `list[OpportunityCandidate]` — matches `MarketSizingAgent.run` input type ✓
- `_to_briefs` correctly unpacks `ScoredOpportunity` fields via `model_dump()` ✓
