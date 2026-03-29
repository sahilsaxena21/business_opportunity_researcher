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
