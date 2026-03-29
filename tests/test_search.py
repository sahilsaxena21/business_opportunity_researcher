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
