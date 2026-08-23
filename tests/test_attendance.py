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


@pytest.mark.asyncio
async def test_identify_falls_back_to_db_when_cache_unavailable(
    client_with_raising_cache, sample_embedding
):
    """When the cache is down, cooldown checks fall back to the DB's
    unique constraint instead of raising -- the first identify still
    succeeds, and a second one within the same day is still deduped
    (via the IntegrityError path), not double-marked.
    """
    await _register(client_with_raising_cache, sample_embedding)

    first = await client_with_raising_cache.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    assert first.status_code == 200
    assert first.json()["status"] == "success"

    second = await client_with_raising_cache.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    assert second.status_code == 200
    assert second.json()["status"] == "ignored"


@pytest.mark.asyncio
async def test_history_rejects_invalid_date_format(client):
    resp = await client.get("/attendance/history", params={"date": "not-a-date"})
    assert resp.status_code == 400
    assert "Invalid date format" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_history_filters_by_valid_date(client, sample_embedding):
    await _register(client, sample_embedding)
    await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )

    import datetime

    today = datetime.date.today().isoformat()  # noqa: DTZ011
    resp = await client.get("/attendance/history", params={"date": today})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp_empty = await client.get("/attendance/history", params={"date": "2000-01-01"})
    assert resp_empty.status_code == 200
    assert resp_empty.json() == []


@pytest.mark.asyncio
async def test_history_pagination_skip_and_limit(
    client, sample_embedding, sample_embedding_far
):
    await _register(client, sample_embedding, employee_id="EMP-100", name="Carol")
    await _register(client, sample_embedding_far, employee_id="EMP-200", name="Dave")
    await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding, "camera_id": "cam-1"},
    )
    await client.post(
        "/attendance/identify",
        json={"embedding": sample_embedding_far, "camera_id": "cam-1"},
    )

    full = await client.get("/attendance/history")
    assert len(full.json()) == 2

    first_page = await client.get("/attendance/history", params={"limit": 1})
    assert len(first_page.json()) == 1

    second_page = await client.get(
        "/attendance/history", params={"skip": 1, "limit": 1}
    )
    assert len(second_page.json()) == 1
    assert first_page.json()[0]["id"] != second_page.json()[0]["id"]


@pytest.mark.asyncio
async def test_history_limit_is_capped(client):
    resp = await client.get("/attendance/history", params={"limit": 999999})
    assert resp.status_code == 422
