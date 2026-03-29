import anthropic
from src import config


class BaseAgent:
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Single call to Claude, returns text content."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def call_with_search(self, system_prompt: str, user_prompt: str, search_fn, max_searches: int = 5) -> str:
        """Tool-use loop: Claude calls search_fn until it returns end_turn."""
        tools = [
            {
                "name": "web_search",
                "description": "Search the web for information. Use for market data, company info, trends.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]

        messages = [{"role": "user", "content": user_prompt}]
        searches_used = 0

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if searches_used >= max_searches:
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "web_search":
                    result = search_fn(block.input["query"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    searches_used += 1

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
