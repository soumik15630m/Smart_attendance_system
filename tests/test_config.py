import importlib

import pytest


def _reload_config():
    """Reimport src.config so module-level validation reruns with current env."""
    import src.config

    return importlib.reload(src.config)


def test_cors_wildcard_when_debug_and_no_origins(monkeypatch):
    from src.config import Settings

    s = Settings(DEBUG=True, ALLOWED_ORIGINS="")
    assert s.cors_origins == ["*"]


def test_cors_parses_comma_separated_origins():
    from src.config import Settings

    s = Settings(ALLOWED_ORIGINS="http://a.com, http://b.com")
    assert s.cors_origins == ["http://a.com", "http://b.com"]


def test_raises_when_debug_false_and_api_key_missing(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("SECRET_KEY", "a-real-secret")
    with pytest.raises(RuntimeError, match="API_KEY"):
        _reload_config()


def test_raises_when_debug_false_and_secret_key_default(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("API_KEY", "some-key")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _reload_config()


def test_no_raise_when_debug_true_regardless_of_keys(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("API_KEY", "")
    _reload_config()  # should not raise
