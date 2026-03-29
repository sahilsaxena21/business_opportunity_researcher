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
