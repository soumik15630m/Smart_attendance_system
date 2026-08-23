"""Tests for the enforce_local_only_mode middleware in src/main.py.

conftest.py sets LOCAL_ONLY=false for every test via the `client` fixture,
so the 403 block path has never actually run. These tests exercise it
directly against a real request/response cycle, spoofing the client host
via httpx's ASGITransport.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.config import settings
from src.main import app


@pytest.mark.asyncio
async def test_blocks_non_loopback_client_when_local_only(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    transport = ASGITransport(app=app, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 403
    assert "Local-only mode" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_allows_loopback_client_when_local_only(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    transport = ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_allows_non_loopback_client_when_not_local_only(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_ONLY", False)
    transport = ASGITransport(app=app, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200


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
def test_is_loopback_host(host, expected):
    from src.main import _is_loopback_host

    assert _is_loopback_host(host) is expected
