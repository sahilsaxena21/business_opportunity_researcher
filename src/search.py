import json
from typing import Callable


def make_search_fn(tavily_client, max_results: int = 5) -> Callable[[str], str]:
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
