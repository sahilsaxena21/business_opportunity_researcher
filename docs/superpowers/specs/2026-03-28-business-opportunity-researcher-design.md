# Business Opportunity Researcher — Design Spec
**Date:** 2026-03-28
**Status:** Approved

---

## Overview

An on-demand multi-agent system that proactively discovers and validates lucrative business opportunities, ranked by their fit with Sahil's unique skillset and experience. The system follows the opportunity validation framework from *The Founder's Playbook* (steps.txt): find a billion-dollar desperate market, confirm a gap in existing solutions, validate business model leverage, and score fit against the founder's unique edge.

---

## Agent Roster

Eight agents, each with a single responsibility:

| Agent | Responsibility | Input | Output |
|---|---|---|---|
| **Orchestrator** | Runs the full pipeline, dispatches agents, manages the 3-round refinement loop, patches gaps between rounds | User trigger + Critic feedback | Final report |
| **Consulting Trend Agent** | Fetches and extracts market trends from McKinsey, BCG, Bain, Deloitte, and similar consulting firm reports | Derived search queries | Raw trend list (trend, source, industry) |
| **Web Signal Agent** | Scans web for emerging opportunity signals — VC investment activity, job posting spikes, Reddit/HN threads, Product Hunt launches, news | Derived search queries | Raw signal list (signal, source, strength) |
| **Market Sizing Agent** | Calculates TAM for each opportunity candidate using the formula: `N customers × $price/yr > $1B` | Opportunity candidate | TAM estimate + pass/fail verdict |
| **Gap Analyst Agent** | Researches existing solutions for each candidate, identifies what is missing or underserved | Opportunity candidate | Gap summary + competitor landscape |
| **Skill Fit Scorer Agent** | Scores each candidate against Sahil's full profile in two dimensions: (1) current fit based on existing skills, (2) reachable fit accounting for skills that could be learned. Opportunities are never excluded solely due to a skill gap — learning effort is estimated instead. | Opportunity candidate + Sahil's profile | Current fit score (0–10) + reachable fit score (0–10) + skills to learn + learning effort (low/medium/high) |
| **Report Writer Agent** | Synthesizes all agent outputs into a ranked shortlist and top-5 deep-dive opportunity briefs | All agent outputs | Draft markdown report |
| **Critic Agent** | Reviews the draft report for weak reasoning, unsupported claims, shallow gap analysis, or poorly justified fit scores. Returns structured critique for the Orchestrator to act on | Draft report | Structured critique + list of gaps to address |

---

## Pipeline Architecture

```
User triggers run
        │
        ▼
  Orchestrator
  - Loads Sahil's profile
  - Derives research queries from profile + industries
        │
        ├──[parallel]──► Consulting Trend Agent → trend_candidates[]
        └──[parallel]──► Web Signal Agent       → signal_candidates[]
                                │
                                ▼
                   Orchestrator merges & deduplicates
                   → opportunity_candidates[] (raw list)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
             Market Sizing  Gap Analyst  Skill Fit Scorer
             (each receives all candidates)
                    │           │           │
                    └───────────┴───────────┘
                                │
                                ▼
                   Orchestrator merges per-opportunity scores
                                │
                    ┌───────────────────────┐
                    │   Refinement Loop ×3  │
                    │                       │
                    │   Report Writer       │
                    │        ▼              │
                    │   Critic Agent        │
                    │        ▼              │
                    │   Orchestrator        │
                    │   patches gaps        │
                    └───────────────────────┘
                                │
                                ▼
                     Final Report (markdown)
```

### Refinement Loop Detail

- Exactly **3 rounds** — no early exit, no extension.
- Each round: Report Writer produces a draft → Critic reviews → Orchestrator patches (re-dispatches specific agents for flagged gaps if needed) → next round begins.
- If a Critic round finds nothing to improve, it returns "no changes needed" and the draft carries forward unchanged.
- After round 3, the final draft is written to output regardless of remaining critique.

---

## Data Flow

1. **Profile loading**: Sahil's profile (skills, domain experience, industries) loaded from `memory/user_profile.md` at runtime.
2. **Query generation**: Orchestrator derives targeted search queries from the profile — e.g. queries anchored to fraud detection, personalization, agentic AI, oil & gas, supply chain.
3. **Research phase**: Consulting Trend Agent and Web Signal Agent run in parallel via `asyncio.gather()`.
4. **Candidate merging**: Orchestrator deduplicates overlapping candidates from both sources, producing a unified `opportunity_candidates[]` list.
5. **Analysis phase**: Market Sizing, Gap Analyst, and Skill Fit Scorer each process the full candidates list. These run sequentially (each agent is already specialized; parallelism is handled at the research phase).
6. **Score merging**: Orchestrator merges per-opportunity scores into a single enriched candidates object.
7. **Refinement loop**: 3× Report Writer → Critic → Orchestrator patch cycle.
8. **Output**: Final markdown report saved to `output/report_YYYY-MM-DD_HH-MM.md`.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Language | Python | Sahil's primary language |
| LLM | `claude-sonnet-4-6` via Anthropic SDK | All agents are Claude API calls with specialized system prompts |
| Web search | `WebSearch` tool (built-in) | Available to research agents for live web queries |
| Orchestration | Pure Python — no LangGraph/CrewAI | Simpler, more debuggable, no framework overhead for V1 |
| Parallelism | `asyncio` + `asyncio.gather()` | For the two research agents running simultaneously |
| Profile input | `memory/user_profile.md` loaded at runtime | Single source of truth for Sahil's skills and experience |
| Output | Markdown file in `output/` directory | Human-readable, easy to version |

---

## Error Handling

- **Agent failure**: If any agent fails (API error, timeout), the Orchestrator logs the failure, skips that agent's contribution, and continues. Affected opportunities are flagged (e.g. "TAM unverified", "Gap analysis unavailable") in the final report rather than being dropped.
- **Empty research results**: If both research agents return zero candidates, the Orchestrator halts and returns: "No opportunities found — try broadening search queries."
- **Refinement loop**: Always runs exactly 3 rounds. A round with no actionable critique passes the draft forward unchanged.
- **Rate limiting**: Small async delay between agent dispatches to avoid Claude API rate limits.

---

## Output Format

### Part 1 — Ranked Shortlist

A table of all discovered opportunities, ranked by a composite score (TAM × gap strength × skill fit):

| Rank | Opportunity | Industry | TAM | Gap Strength | Skill Fit (0–10) | Leverage Model |
|---|---|---|---|---|---|---|
| 1 | ... | ... | $XB | Strong | 8.5 | SaaS |

### Part 2 — Top 5 Deep-Dive Briefs

One section per top opportunity:

```
## Opportunity: [Name]

### The Trend
[What market shift is happening, with source citation]

### The Gap
[What existing solutions fail to do]

### Market Size
[TAM math: N customers × $X/yr = $YB]

### Your Edge
[Specific skills and experiences from Sahil's profile that apply, with explicit references]

### Recommended Business Model
[How to structure as a high-leverage business: recurring revenue, 70%+ margins, tech-scales, own product]

### Confidence Score
[Critic's final assessment after 3 refinement rounds, with brief rationale]
```

---

## Sahil's Profile (Skill Fit Scoring Basis)

The Skill Fit Scorer uses the following domains when evaluating opportunity fit:

**Data Science & AI**
- Agentic AI pipelines (production LLM multi-step reasoning)
- ML-driven personalization and behavioral propensity modeling
- Fraud detection (clickstream, XGBoost, behavioral feature engineering)
- Demand forecasting and supply chain optimization
- Real-time anomaly detection (IoT)
- Full MLOps stack (GCP, AWS, Docker, Airflow)

**Industry Domain Knowledge**
- Banking / Fintech (Scotiabank — fraud, personalization, agentic AI)
- Retail / Supply Chain (Loblaw — forecasting, inventory)
- IoT / HVAC (7Senses — real-time systems)
- Oil & Gas / EPCM (Teng Consulting — Suncor, IOL — mechanical design, procurement, project delivery)

**Business & Execution**
- Translating ambiguous business problems into scoped technical solutions
- Quantifying ROI and presenting to executives
- Procurement, vendor management, RFQs, bid evaluations
- Cross-functional delivery at scale ($40M+ in projects)

---

## Constraints & Scope

- **V1 scope**: On-demand runs only. Scheduled/recurring runs are out of scope.
- **No UI**: CLI trigger only for V1. A web interface is a future consideration.
- **No persistent memory between runs**: Each run is stateless. Output reports are the only artifact.
- **Opportunity validation only**: The system surfaces and scores opportunities. It does not build business plans, financial models, or go-to-market strategies.
