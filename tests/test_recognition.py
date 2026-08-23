"""Tests for src/services/recognition.py.

test_persons.py and test_attendance.py already cover "nobody registered"
and "exact match" through the API. This file targets the edge cases that
were never exercised: an inactive person is excluded from matching even
though they're the closest embedding, and a registered face just outside
SIMILARITY_THRESHOLD is correctly treated as no match.
"""

import math

import pytest

from src.config import settings
from src.models.person import Person
from src.services.recognition import RecognitionService


def _embedding_at_cosine_similarity(similarity: float) -> list[float]:
    """Build a 512-dim unit vector whose cosine similarity to
    sample_embedding ([1, 0, 0, ...]) is exactly `similarity`.
    """
    vec = [0.0] * 512
    vec[0] = similarity
    vec[1] = math.sqrt(max(0.0, 1.0 - similarity**2))
    return vec


async def _add_person(db_session, embedding, is_active=True, employee_id="EMP-X"):
    person = Person(
        name="Test Person",
        employee_id=employee_id,
        role="employee",
        embedding=embedding,
        is_active=is_active,
    )
    db_session.add(person)
    await db_session.flush()
    return person


@pytest.mark.asyncio
async def test_inactive_person_is_excluded_from_matching(db_session, sample_embedding):
    await _add_person(db_session, sample_embedding, is_active=False)

    service = RecognitionService(db_session)
    match = await service.find_nearest_match(sample_embedding)

    assert match is None


@pytest.mark.asyncio
async def test_active_person_beyond_threshold_is_no_match(db_session, sample_embedding):
    # cosine distance = 1 - similarity, so similarity just below the
    # threshold boundary (1 - SIMILARITY_THRESHOLD) puts distance just
    # *over* the threshold -- a near miss, not a match.
    boundary_similarity = 1.0 - settings.SIMILARITY_THRESHOLD
    far_but_close_embedding = _embedding_at_cosine_similarity(boundary_similarity - 0.1)
    await _add_person(db_session, far_but_close_embedding, is_active=True)

    service = RecognitionService(db_session)
    match = await service.find_nearest_match(sample_embedding)

    assert match is None


@pytest.mark.asyncio
async def test_active_person_just_within_threshold_is_a_match(
    db_session, sample_embedding
):
    boundary_similarity = 1.0 - settings.SIMILARITY_THRESHOLD
    just_within_embedding = _embedding_at_cosine_similarity(boundary_similarity + 0.1)
    person = await _add_person(db_session, just_within_embedding, is_active=True)

    service = RecognitionService(db_session)
    match = await service.find_nearest_match(sample_embedding)

    assert match is not None
    matched_person, distance = match
    assert matched_person.id == person.id
    assert distance < settings.SIMILARITY_THRESHOLD


@pytest.mark.asyncio
async def test_picks_nearest_active_person_over_inactive_closer_one(
    db_session, sample_embedding
):
    # The inactive person is a near-exact match (very close); the active
    # person is farther but still within threshold. The service must
    # still return the active one, not silently prefer the closer match.
    await _add_person(
        db_session, sample_embedding, is_active=False, employee_id="EMP-INACTIVE"
    )
    boundary_similarity = 1.0 - settings.SIMILARITY_THRESHOLD
    active_embedding = _embedding_at_cosine_similarity(boundary_similarity + 0.1)
    active_person = await _add_person(
        db_session, active_embedding, is_active=True, employee_id="EMP-ACTIVE"
    )

    service = RecognitionService(db_session)
    match = await service.find_nearest_match(sample_embedding)

    assert match is not None
    matched_person, _distance = match
    assert matched_person.id == active_person.id
