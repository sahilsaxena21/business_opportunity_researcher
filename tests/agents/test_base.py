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
