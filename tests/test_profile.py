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
