import pytest


async def _register(client, embedding, employee_id="EMP-100", name="Carol"):
    resp = await client.post(
        "/persons/register",
        json={
            "name": name,
            "employee_id": employee_id,
            "role": "employee",
            "is_active": True,
            "embedding": embedding,
        },
    )
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_identify_unknown_embedding(client, sample_embedding):
    resp = await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown"


@pytest.mark.asyncio
async def test_identify_marks_attendance_for_known_face(client, sample_embedding):
    await _register(client, sample_embedding)
    resp = await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["person_name"] == "Carol"


@pytest.mark.asyncio
async def test_identify_ignores_repeat_within_cooldown(client, sample_embedding):
    await _register(client, sample_embedding)
    first = await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    second = await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    assert first.json()["status"] == "success"
    assert second.json()["status"] == "ignored"


@pytest.mark.asyncio
async def test_history_returns_marked_attendance(client, sample_embedding):
    await _register(client, sample_embedding)
    await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    resp = await client.get("/attendance/history")
    assert resp.status_code == 200
    records = resp.json()
    assert len(records) == 1
    assert records[0]["person"]["name"] == "Carol"
