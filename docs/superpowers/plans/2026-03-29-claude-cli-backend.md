# Claude CLI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct Anthropic SDK calls in `BaseAgent` with Claude CLI subprocess calls so the app runs under the user's Claude Pro subscription with no separate API billing.

**Architecture:** `BaseAgent._cli_call()` shells out to `claude -p` with `--output-format json` for all LLM calls. `call()` delegates directly to `_cli_call`. `call_with_search()` implements a dynamic loop using a text protocol (`SEARCH: <query>`) so Claude can request Tavily searches iteratively before producing its final answer. No agent subclass changes.

**Tech Stack:** Python 3.9+, `subprocess`, `claude` CLI (Claude Code), `tavily-python`, `pydantic`, `pytest`

---

## File Map

| File | Change |
|---|---|
| `src/agents/base.py` | Full rewrite — drop `anthropic` import and client, add `_cli_call`, rewrite `call` and `call_with_search` |
| `src/agents/orchestrator.py` | Drop `anthropic_client` param from `__init__` and `Orchestrator` class |
| `main.py` | Drop `anthropic` import and `anthropic.Anthropic()` instantiation |
| `src/config.py` | Drop `ANTHROPIC_API_KEY` |
| `.env` | Remove `ANTHROPIC_API_KEY` line |
| `requirements.txt` | Remove `anthropic>=0.40.0` |
| `tests/agents/test_consulting_trend.py` | Replace `make_mock_client` pattern with `patch.object(_cli_call)` |
| `tests/agents/test_web_signal.py` | Same |
| `tests/agents/test_market_sizing.py` | Same |
| `tests/agents/test_gap_analyst.py` | Same |
| `tests/agents/test_skill_fit_scorer.py` | Same |
| `tests/agents/test_report_writer.py` | Same |
| `tests/agents/test_critic.py` | Same |
| `tests/agents/test_orchestrator.py` | Drop `anthropic_client=MagicMock()` from `Orchestrator(...)` call |

---

### Task 1: Rewrite `BaseAgent`

**Files:**
- Modify: `src/agents/base.py`

- [ ] **Step 1: Write the failing tests for `_cli_call` and `call`**

Replace the entire content of `tests/agents/test_base.py` (create it if it doesn't exist):

```python
# tests/agents/test_base.py
import pytest
from unittest.mock import patch, MagicMock
from src.agents.base import BaseAgent


class ConcreteAgent(BaseAgent):
    pass


def test_call_delegates_to_cli_call():
    agent = ConcreteAgent()
    with patch.object(agent, '_cli_call', return_value="hello") as mock:
        result = agent.call("sys", "usr")
    mock.assert_called_once_with("sys", "usr")
    assert result == "hello"


def test_cli_call_parses_json_result(monkeypatch):
    import subprocess, json
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = json.dumps({"type": "result", "result": "output text"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake)
    agent = ConcreteAgent()
    assert agent._cli_call("sys", "usr") == "output text"


def test_cli_call_raises_on_nonzero(monkeypatch):
    import subprocess
    fake = MagicMock()
    fake.returncode = 1
    fake.stderr = "some error"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake)
    agent = ConcreteAgent()
    with pytest.raises(RuntimeError, match="Claude CLI error"):
        agent._cli_call("sys", "usr")


def test_call_with_search_returns_on_end_turn():
    agent = ConcreteAgent()
    responses = iter(["final answer here"])
    with patch.object(agent, '_cli_call', side_effect=lambda s, u: next(responses)):
        result = agent.call_with_search("sys", "usr", lambda q: "[]")
    assert result == "final answer here"


def test_call_with_search_executes_one_search():
    agent = ConcreteAgent()
    search_called = []

    def fake_search(q):
        search_called.append(q)
        return '["result"]'

    responses = iter(["SEARCH: fintech trends 2025", "final answer"])
    with patch.object(agent, '_cli_call', side_effect=lambda s, u: next(responses)):
        result = agent.call_with_search("sys", "usr", fake_search)

    assert len(search_called) == 1
    assert search_called[0] == "fintech trends 2025"
    assert result == "final answer"


def test_call_with_search_stops_at_max_searches():
    agent = ConcreteAgent()
    call_count = [0]

    def always_search(s, u):
        call_count[0] += 1
        return "SEARCH: query"

    with patch.object(agent, '_cli_call', side_effect=always_search):
        # max_searches=2 means 2 search rounds + 1 final forced call = 3 total _cli_calls
        agent.call_with_search("sys", "usr", lambda q: "[]", max_searches=2)

    assert call_count[0] == 3  # 2 search loops + 1 forced final
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
py -3.9 -m pytest tests/agents/test_base.py -v
```

Expected: Most tests FAIL (BaseAgent still uses anthropic client)

- [ ] **Step 3: Rewrite `src/agents/base.py`**

Replace the entire file with:

```python
import json as _json
import subprocess
from src import config

_SEARCH_INSTRUCTION = (
    "\n\nYou have web search access. To search, output ONLY this line and nothing else:\n"
    "SEARCH: <your query>\n"
    "When you have enough information, output your final answer normally."
)


class BaseAgent:
    def __init__(self, model: str = config.MODEL):
        self.model = model

    def _cli_call(self, system: str, user: str) -> str:
        """Shell out to the Claude CLI and return the text result."""
        result = subprocess.run(
            [
                "claude", "-p", user,
                "--system", system,
                "--model", self.model,
                "--output-format", "json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr.strip()}")
        data = _json.loads(result.stdout)
        return data.get("result", "")

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Single call to Claude, returns text content."""
        return self._cli_call(system_prompt, user_prompt)

    def call_with_search(
        self,
        system_prompt: str,
        user_prompt: str,
        search_fn,
        max_searches: int = 5,
    ) -> str:
        """Dynamic search loop: Claude issues SEARCH: requests until it has enough info."""
        search_results = []
        searches_used = 0

        while searches_used < max_searches:
            context = ""
            if search_results:
                context = "\n\nSearch results so far:\n" + "\n\n".join(
                    f"[Search {i + 1}: {q}]\n{r}"
                    for i, (q, r) in enumerate(search_results)
                )

            response = self._cli_call(
                system_prompt + _SEARCH_INSTRUCTION,
                user_prompt + context,
            )

            if response.strip().upper().startswith("SEARCH:"):
                query = response.strip()[7:].strip()
                result = search_fn(query)
                search_results.append((query, result))
                searches_used += 1
            else:
                return response

        # max_searches reached — force final answer without search option
        context = "\n\nSearch results:\n" + "\n\n".join(
            f"[Search {i + 1}: {q}]\n{r}"
            for i, (q, r) in enumerate(search_results)
        )
        return self._cli_call(system_prompt, user_prompt + context)
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
py -3.9 -m pytest tests/agents/test_base.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py tests/agents/test_base.py
git commit -m "feat: replace anthropic client with claude CLI subprocess in BaseAgent"
```

---

### Task 2: Update all agent tests (drop `make_mock_client`)

**Files:**
- Modify: `tests/agents/test_consulting_trend.py`
- Modify: `tests/agents/test_web_signal.py`
- Modify: `tests/agents/test_market_sizing.py`
- Modify: `tests/agents/test_gap_analyst.py`
- Modify: `tests/agents/test_skill_fit_scorer.py`
- Modify: `tests/agents/test_report_writer.py`
- Modify: `tests/agents/test_critic.py`

- [ ] **Step 1: Run existing agent tests — confirm they now fail**

```bash
py -3.9 -m pytest tests/agents/ -v --ignore=tests/agents/test_base.py --ignore=tests/agents/test_orchestrator.py
```

Expected: FAIL — agents no longer accept a client argument

- [ ] **Step 2: Rewrite `tests/agents/test_consulting_trend.py`**

```python
# tests/agents/test_consulting_trend.py
import pytest
from unittest.mock import patch
from src.agents.consulting_trend import ConsultingTrendAgent
from src.models.schemas import TrendCandidate

SAMPLE_JSON = '''[
  {"title": "AI-Driven Fraud Detection", "description": "Banks adopting ML", "industry": "fintech", "source": "McKinsey"},
  {"title": "Supply Chain AI", "description": "AI optimizing logistics", "industry": "retail", "source": "BCG"}
]'''


def test_returns_list():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("Data scientist profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_trend_candidates():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("Data scientist profile", lambda q: "[]")
    assert all(isinstance(r, TrendCandidate) for r in results)


def test_parses_title():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = ConsultingTrendAgent()
        results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "AI-Driven Fraud Detection" for r in results)


def test_returns_empty_on_bad_json():
    with patch.object(ConsultingTrendAgent, '_cli_call', return_value="Sorry, no trends found."):
        agent = ConsultingTrendAgent()
        results = agent.run("profile", lambda q: "[]")
    assert results == []
```

- [ ] **Step 3: Rewrite `tests/agents/test_web_signal.py`**

```python
# tests/agents/test_web_signal.py
import pytest
from unittest.mock import patch
from src.agents.web_signal import WebSignalAgent
from src.models.schemas import SignalCandidate

SAMPLE_JSON = '''[
  {"title": "VC surge in fraud AI", "description": "Heavy investment", "signal_type": "vc_investment", "source": "TechCrunch", "strength": "strong"},
  {"title": "Job spike in MLOps", "description": "Growing demand", "signal_type": "job_spike", "source": "LinkedIn", "strength": "moderate"}
]'''


def test_returns_list():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert isinstance(results, list)


def test_returns_signal_candidates():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert all(isinstance(r, SignalCandidate) for r in results)


def test_parses_title():
    with patch.object(WebSignalAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert any(r.title == "VC surge in fraud AI" for r in results)


def test_returns_empty_on_bad_json():
    with patch.object(WebSignalAgent, '_cli_call', return_value="No signals."):
        agent = WebSignalAgent()
        results = agent.run("profile", lambda q: "[]")
    assert results == []
```

- [ ] **Step 4: Rewrite `tests/agents/test_market_sizing.py`**

```python
# tests/agents/test_market_sizing.py
import json
import pytest
from unittest.mock import patch
from src.agents.market_sizing import MarketSizingAgent
from src.models.schemas import OpportunityCandidate, SizedOpportunity

CANDIDATE = OpportunityCandidate(
    title="AI Fraud Detection",
    description="ML-based fraud platform",
    industry="fintech",
    sources=["McKinsey"],
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud platform",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k/yr",
    "tam_passes": True,
}])


def test_returns_list():
    with patch.object(MarketSizingAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert isinstance(results, list)


def test_returns_sized_opportunities():
    with patch.object(MarketSizingAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert all(isinstance(r, SizedOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = MarketSizingAgent()
    assert agent.run([]) == []


def test_returns_empty_on_bad_json():
    with patch.object(MarketSizingAgent, '_cli_call', return_value="not json"):
        agent = MarketSizingAgent()
        results = agent.run([CANDIDATE])
    assert results == []
```

- [ ] **Step 5: Rewrite `tests/agents/test_gap_analyst.py`**

```python
# tests/agents/test_gap_analyst.py
import json
import pytest
from unittest.mock import patch
from src.agents.gap_analyst import GapAnalystAgent
from src.models.schemas import SizedOpportunity, GapAnalyzedOpportunity

CANDIDATE = SizedOpportunity(
    title="AI Fraud Detection",
    description="ML-based fraud",
    industry="fintech",
    sources=["McKinsey"],
    tam_estimate="$4.2B",
    tam_math="5000 banks × $840k",
    tam_passes=True,
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k",
    "tam_passes": True,
    "gap_summary": "Legacy rules-based systems miss novel attacks",
    "existing_solutions": ["Featurespace", "Sardine"],
    "key_gap": "Real-time adaptive ML for novel fraud patterns",
}])


def test_returns_list():
    with patch.object(GapAnalystAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = GapAnalystAgent()
        results = agent.run([CANDIDATE])
    assert isinstance(results, list)


def test_returns_gap_analyzed():
    with patch.object(GapAnalystAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = GapAnalystAgent()
        results = agent.run([CANDIDATE])
    assert all(isinstance(r, GapAnalyzedOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = GapAnalystAgent()
    assert agent.run([]) == []
```

- [ ] **Step 6: Rewrite `tests/agents/test_skill_fit_scorer.py`**

```python
# tests/agents/test_skill_fit_scorer.py
import json
import pytest
from unittest.mock import patch
from src.agents.skill_fit_scorer import SkillFitScorerAgent
from src.models.schemas import GapAnalyzedOpportunity, ScoredOpportunity

CANDIDATE = GapAnalyzedOpportunity(
    title="AI Fraud Detection",
    description="ML-based fraud",
    industry="fintech",
    sources=["McKinsey"],
    tam_estimate="$4.2B",
    tam_math="5000 banks × $840k",
    tam_passes=True,
    gap_summary="Legacy systems miss novel attacks",
    existing_solutions=["Featurespace"],
    key_gap="Real-time adaptive ML",
)

SAMPLE_JSON = json.dumps([{
    "title": "AI Fraud Detection",
    "description": "ML-based fraud",
    "industry": "fintech",
    "sources": ["McKinsey"],
    "tam_estimate": "$4.2B",
    "tam_math": "5000 banks × $840k",
    "tam_passes": True,
    "gap_summary": "Legacy systems miss novel attacks",
    "existing_solutions": ["Featurespace"],
    "key_gap": "Real-time adaptive ML",
    "current_fit_score": 8.5,
    "reachable_fit_score": 9.0,
    "fit_rationale": "Strong fraud ML background",
    "relevant_skills": ["XGBoost", "behavioral ML"],
    "skills_to_learn": ["sales"],
    "learning_effort": "low",
}])

PROFILE = {"raw": "Data scientist with fraud detection experience"}


def test_returns_list():
    with patch.object(SkillFitScorerAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = SkillFitScorerAgent()
        results = agent.run([CANDIDATE], PROFILE)
    assert isinstance(results, list)


def test_returns_scored_opportunities():
    with patch.object(SkillFitScorerAgent, '_cli_call', return_value=SAMPLE_JSON):
        agent = SkillFitScorerAgent()
        results = agent.run([CANDIDATE], PROFILE)
    assert all(isinstance(r, ScoredOpportunity) for r in results)


def test_returns_empty_for_empty_input():
    agent = SkillFitScorerAgent()
    assert agent.run([], PROFILE) == []
```

- [ ] **Step 7: Rewrite `tests/agents/test_report_writer.py`**

```python
# tests/agents/test_report_writer.py
import pytest
from unittest.mock import patch
from src.agents.report_writer import ReportWriterAgent
from src.models.schemas import ScoredOpportunity

OPPORTUNITY = ScoredOpportunity(
    title="AI Fraud Detection",
    description="ML-based fraud",
    industry="fintech",
    sources=["McKinsey"],
    tam_estimate="$4.2B",
    tam_math="5000 banks × $840k",
    tam_passes=True,
    gap_summary="Legacy systems miss novel attacks",
    existing_solutions=["Featurespace"],
    key_gap="Real-time adaptive ML",
    current_fit_score=8.5,
    reachable_fit_score=9.0,
    fit_rationale="Strong fraud ML background",
    relevant_skills=["XGBoost"],
    skills_to_learn=["sales"],
    learning_effort="low",
)

MARKDOWN = "# Business Opportunity Report\n\n## Top Opportunities\n..."


def test_returns_string():
    with patch.object(ReportWriterAgent, '_cli_call', return_value=MARKDOWN):
        agent = ReportWriterAgent()
        result = agent.run([OPPORTUNITY], "")
    assert isinstance(result, str)


def test_returns_markdown_content():
    with patch.object(ReportWriterAgent, '_cli_call', return_value=MARKDOWN):
        agent = ReportWriterAgent()
        result = agent.run([OPPORTUNITY], "")
    assert "Business Opportunity Report" in result


def test_incorporates_critique():
    captured = []

    def fake_cli(system, user):
        captured.append(user)
        return MARKDOWN

    with patch.object(ReportWriterAgent, '_cli_call', side_effect=fake_cli):
        agent = ReportWriterAgent()
        agent.run([OPPORTUNITY], "TAM estimate lacks citation")

    assert any("TAM estimate lacks citation" in u for u in captured)
```

- [ ] **Step 8: Rewrite `tests/agents/test_critic.py`**

```python
# tests/agents/test_critic.py
import json
import pytest
from unittest.mock import patch
from src.agents.critic import CriticAgent
from src.models.schemas import CritiqueResult

GOOD_JSON = json.dumps({
    "round_number": 1,
    "has_changes": True,
    "critiques": [{"opportunity_title": "AI Fraud", "issue": "TAM unsupported", "suggested_improvement": "Cite source"}],
    "overall_quality": "fair",
})

NO_CHANGE_JSON = json.dumps({
    "round_number": 2,
    "has_changes": False,
    "critiques": [],
    "overall_quality": "excellent",
})


def test_returns_critique_result():
    with patch.object(CriticAgent, '_cli_call', return_value=GOOD_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert isinstance(result, CritiqueResult)


def test_parses_has_changes():
    with patch.object(CriticAgent, '_cli_call', return_value=GOOD_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert result.has_changes is True


def test_parses_no_changes():
    with patch.object(CriticAgent, '_cli_call', return_value=NO_CHANGE_JSON):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=2)
    assert result.has_changes is False
    assert result.critiques == []


def test_returns_default_on_bad_json():
    with patch.object(CriticAgent, '_cli_call', return_value="not json"):
        agent = CriticAgent()
        result = agent.run("# Report", round_number=1)
    assert isinstance(result, CritiqueResult)
    assert result.has_changes is False
```

- [ ] **Step 9: Run all updated agent tests — confirm they pass**

```bash
py -3.9 -m pytest tests/agents/ -v --ignore=tests/agents/test_orchestrator.py
```

Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add tests/agents/
git commit -m "test: update agent tests to mock _cli_call instead of anthropic client"
```

---

### Task 3: Update `Orchestrator` and its test

**Files:**
- Modify: `src/agents/orchestrator.py`
- Modify: `tests/agents/test_orchestrator.py`

- [ ] **Step 1: Write the failing orchestrator init test**

Replace `test_orchestrator_initializes` in `tests/agents/test_orchestrator.py`:

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
    orch = Orchestrator(tavily_client=MagicMock())
    assert orch is not None
```

- [ ] **Step 2: Run — confirm `test_orchestrator_initializes` fails**

```bash
py -3.9 -m pytest tests/agents/test_orchestrator.py::test_orchestrator_initializes -v
```

Expected: FAIL — `Orchestrator.__init__` still requires `anthropic_client`

- [ ] **Step 3: Update `src/agents/orchestrator.py`**

Remove the `anthropic_client` parameter and all `import anthropic` references. Change only the `__init__` method and imports:

```python
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
```

And update `Orchestrator.__init__`:

```python
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
```

Keep all other methods (`run`, `_research_parallel`, `_refinement_loop`, `_save`) unchanged.

- [ ] **Step 4: Run orchestrator tests — confirm all pass**

```bash
py -3.9 -m pytest tests/agents/test_orchestrator.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "feat: remove anthropic_client from Orchestrator"
```

---

### Task 4: Update `main.py`, `config.py`, `.env`, `requirements.txt`

**Files:**
- Modify: `main.py`
- Modify: `src/config.py`
- Modify: `.env`
- Modify: `requirements.txt`

- [ ] **Step 1: Update `main.py`**

Replace the full file:

```python
import sys
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

    tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
    orchestrator = Orchestrator(tavily_client=tavily_client)

    report = orchestrator.run(profile)

    print(f"\nDone. {report.total_opportunities_found} opportunities found.")
    if report.ranked_opportunities:
        print(f"Top opportunity: {report.ranked_opportunities[0].title}")
    print("\n--- Report Preview ---")
    print(report.report_markdown[:600])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update `src/config.py`**

Replace the full file:

```python
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
MODEL = "claude-sonnet-4-6"
REFINEMENT_ROUNDS = 3
PROFILE_PATH = "memory/user_profile.md"
OUTPUT_DIR = "output"
```

- [ ] **Step 3: Update `.env`**

Remove the `ANTHROPIC_API_KEY` line. The file should contain only:

```
TAVILY_API_KEY=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
```

- [ ] **Step 4: Update `requirements.txt`**

Remove the `anthropic>=0.40.0` line. File should be:

```
pydantic>=2.0.0
tavily-python>=0.3.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 5: Run the full test suite**

```bash
py -3.9 -m pytest tests/ -v
```

Expected: All tests PASS (no anthropic import errors)

- [ ] **Step 6: Commit**

```bash
git add main.py src/config.py .env requirements.txt
git commit -m "feat: remove anthropic dependency from main, config, env, and requirements"
```

---

### Task 5: Smoke test the full run

**Files:** None (runtime verification only)

- [ ] **Step 1: Verify `claude` CLI is on PATH**

```bash
claude --version
```

Expected: prints a version string (e.g. `1.x.x`)

- [ ] **Step 2: Run the app**

```bash
py -3.9 main.py
```

Expected output pattern:
```
=== Business Opportunity Researcher ===

Profile loaded: Sahil Saxena

[1/6] Running research agents in parallel...
      N trends, M signals found
[2/6] Merging candidates...
...
[6/6] Refinement loop (3 rounds)...
...
Done. X opportunities found.
Top opportunity: <title>
```

- [ ] **Step 3: Confirm report file written**

```bash
ls output/
```

Expected: a file named `report_YYYY-MM-DD_HH-MM.md`

- [ ] **Step 4: Commit**

```bash
git add output/.gitkeep 2>/dev/null; git commit -m "chore: verify end-to-end run with claude CLI backend" --allow-empty
```
