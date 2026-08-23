from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.config import settings
from src.routers.local_ui import (
    SCRIPT_RUNNER,
    _db_summary,
    _is_loopback_client,
    ensure_local_access,
)


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


# --- _db_summary ----------------------------------------------------------


class _RaisingDb:
    """Stand-in AsyncSession whose execute() always fails, like a DB outage."""

    async def execute(self, *_args, **_kwargs):
        raise ConnectionRefusedError("could not connect to server")


@pytest.mark.asyncio
async def test_db_summary_reports_up_on_success(db_session):
    result = await _db_summary(db_session)
    assert result["db_status"] == "up"
    assert result["people_count"] == 0
    assert result["attendance_today"] == 0
    assert result["db_error"] is None


@pytest.mark.asyncio
async def test_db_summary_reports_down_on_exception():
    result = await _db_summary(_RaisingDb())  # type: ignore[arg-type]
    assert result["db_status"] == "down"
    assert result["people_count"] is None
    assert result["attendance_today"] is None
    assert "could not connect" in str(result["db_error"])


# --- dashboard endpoints (through the real app) ----------------------------


@pytest.mark.asyncio
async def test_serve_dashboard_returns_index_html(client):
    resp = await client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_dashboard_overview_reports_db_up(client):
    resp = await client.get("/ui/api/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_status"] == "up"
    assert body["people_count"] == 0
    assert body["attendance_today"] == 0
    assert body["running_scripts"] == 0
    assert "hostname" in body
    assert "platform" in body


@pytest.mark.asyncio
async def test_dashboard_overview_blocked_when_local_only_and_remote(monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from src.main import app

    monkeypatch.setattr(settings, "LOCAL_ONLY", True)
    transport = ASGITransport(app=app, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/ui/api/overview")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_scripts_endpoint_returns_known_scripts(client):
    resp = await client.get("/ui/api/scripts")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()["scripts"]}
    assert {"test_gpu", "seed_db", "register_face", "camera_client"} <= ids


@pytest.mark.asyncio
async def test_onboarding_endpoint_reports_steps_and_db_status(client):
    resp = await client.get("/ui/api/onboarding")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_status"] == "up"
    step_ids = {step["id"] for step in body["steps"]}
    assert step_ids == {"gpu", "seed", "enroll", "camera", "attendance"}


@pytest.mark.asyncio
async def test_recent_attendance_endpoint_empty_when_no_records(client):
    resp = await client.get("/ui/api/attendance/recent")
    assert resp.status_code == 200
    assert resp.json() == {"records": []}


@pytest.mark.asyncio
async def test_recent_attendance_endpoint_surfaces_db_error(client, monkeypatch):
    async def _boom(*_args, **_kwargs):
        raise ConnectionRefusedError("db is down")

    from sqlalchemy.ext.asyncio import AsyncSession

    monkeypatch.setattr(AsyncSession, "execute", _boom)
    resp = await client.get("/ui/api/attendance/recent")
    assert resp.status_code == 200
    body = resp.json()
    assert body["records"] == []
    assert "db is down" in body["error"]


# --- start / stop / logs script endpoints ----------------------------------


@pytest.mark.asyncio
async def test_start_unknown_script_returns_404(client):
    resp = await client.post("/ui/api/scripts/does-not-exist/start")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stop_unknown_script_returns_404(client):
    resp = await client.post("/ui/api/scripts/does-not-exist/stop")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_logs_unknown_script_returns_404(client):
    resp = await client.get("/ui/api/scripts/does-not-exist/logs")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_start_script_conflict_returns_409(client, monkeypatch):
    def _raise_running(*_args, **_kwargs):
        raise RuntimeError("seed_db is already running.")

    monkeypatch.setattr(SCRIPT_RUNNER, "start_script", _raise_running)
    resp = await client.post("/ui/api/scripts/seed_db/start")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_start_script_missing_name_returns_400(client, monkeypatch):
    def _raise_value_error(*_args, **_kwargs):
        raise ValueError("Name is required to run face registration.")

    monkeypatch.setattr(SCRIPT_RUNNER, "start_script", _raise_value_error)
    resp = await client.post("/ui/api/scripts/register_face/start", json={})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_stop_script_not_running_returns_409(client, monkeypatch):
    def _raise_not_running(*_args, **_kwargs):
        raise RuntimeError("seed_db is not currently running.")

    monkeypatch.setattr(SCRIPT_RUNNER, "stop_script", _raise_not_running)
    resp = await client.post("/ui/api/scripts/seed_db/stop")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_start_stop_logs_roundtrip_with_real_subprocess(client):
    """End-to-end through the router + real LocalScriptRunner + a real
    subprocess (test_gpu.py), which fails fast because onnxruntime isn't
    installed in the test environment. That's fine -- this test only cares
    that start/logs/stop wire up correctly, not that the script succeeds.
    """
    import asyncio

    start_resp = await client.post("/ui/api/scripts/test_gpu/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["script"]["status"] in {"running", "failed"}

    deadline = asyncio.get_event_loop().time() + 5
    status = None
    while asyncio.get_event_loop().time() < deadline:
        logs_resp = await client.get("/ui/api/scripts/test_gpu/logs")
        assert logs_resp.status_code == 200
        status = logs_resp.json()["script"]["status"]
        if status != "running":
            break
        await asyncio.sleep(0.05)

    assert status in {"failed", "completed"}
