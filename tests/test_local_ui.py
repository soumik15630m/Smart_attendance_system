from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.config import settings
from src.routers.local_ui import _is_loopback_client, ensure_local_access


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("10.0.0.5", False),
        ("8.8.8.8", False),
        (None, False),
        ("not-an-ip", False),
    ],
)
def test_is_loopback_client(host, expected):
    assert _is_loopback_client(host) is expected


def _fake_request(host):
    return SimpleNamespace(client=SimpleNamespace(host=host) if host else None)


def test_ensure_local_access_allows_loopback(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    ensure_local_access(_fake_request("127.0.0.1"))  # should not raise


def test_ensure_local_access_blocks_remote(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    with pytest.raises(HTTPException) as exc_info:
        ensure_local_access(_fake_request("8.8.8.8"))
    assert exc_info.value.status_code == 403


def test_ensure_local_access_skips_check_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", False)
    ensure_local_access(_fake_request("8.8.8.8"))  # should not raise
