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
