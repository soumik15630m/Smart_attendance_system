import pytest


def _person_payload(name="Alice", employee_id="EMP-001", embedding=None):
    return {
        "name": name,
        "employee_id": employee_id,
        "role": "employee",
        "is_active": True,
        "embedding": embedding,
    }


@pytest.mark.asyncio
async def test_register_person_success(client, sample_embedding):
    resp = await client.post(
        "/persons/register", json=_person_payload(embedding=sample_embedding)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Alice"
    assert body["employee_id"] == "EMP-001"


@pytest.mark.asyncio
async def test_register_duplicate_employee_id_rejected(
    client, sample_embedding, sample_embedding_far
):
    await client.post(
        "/persons/register", json=_person_payload(embedding=sample_embedding)
    )
    resp = await client.post(
        "/persons/register",
        json=_person_payload(name="Bob", embedding=sample_embedding_far),
    )
    assert resp.status_code == 400
    assert "already registered" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_register_duplicate_face_rejected(client, sample_embedding):
    await client.post(
        "/persons/register",
        json=_person_payload(employee_id="EMP-001", embedding=sample_embedding),
    )
    resp = await client.post(
        "/persons/register",
        json=_person_payload(
            name="Bob", employee_id="EMP-002", embedding=sample_embedding
        ),
    )
    assert resp.status_code == 400
    assert "already registered" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_register_rejects_wrong_embedding_length(client):
    resp = await client.post(
        "/persons/register", json=_person_payload(embedding=[0.1] * 10)
    )
    assert resp.status_code == 422
