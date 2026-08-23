import pytest


@pytest.mark.asyncio
async def test_liveness_ok(client):
    resp = await client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_db_health_ok(client):
    resp = await client.get("/health/db")
    assert resp.status_code == 200
    assert resp.json() == {"status": "up", "database": "connected"}


@pytest.mark.asyncio
async def test_db_health_returns_503_when_db_down(client, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    async def _boom(self, *_args, **_kwargs):
        raise ConnectionError("could not connect to server")

    monkeypatch.setattr(AsyncSession, "execute", _boom)

    resp = await client.get("/health/db")
    assert resp.status_code == 503
    assert "Database connection failed" in resp.json()["detail"]
