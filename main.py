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
