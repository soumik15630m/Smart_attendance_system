"""Shared pytest fixtures.

Requires a Postgres database with the `vector` extension available
(matches the `pgvector/pgvector:pg16` service used in CI). Point
DATABASE_URL at it before running pytest; defaults to the CI service's
connection string if unset.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/test_db"
)
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOCAL_ONLY", "false")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import engine, get_db
from src.models import (  # noqa: F401 - registers tables on Base.metadata
    Attendance,
    Base,
    Person,
)
from src.redis_config import get_redis


class FakeCache:
    """In-memory stand-in for CacheClient, avoids needing real Redis in tests."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self._store[key] = value

    async def close(self) -> None:
        self._store.clear()


@pytest_asyncio.fixture(scope="session")
async def prepare_database():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(prepare_database):
    """A session bound to a transaction that's rolled back after each test."""
    async with engine.connect() as conn:
        await conn.begin()
        session = AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()


@pytest.fixture
def fake_cache():
    return FakeCache()


class RaisingCache:
    """CacheClient stand-in that always raises, to exercise the
    cache-failure fallback paths in AttendanceService (which are never
    triggered by FakeCache, since it never raises).
    """

    async def get(self, key: str) -> str | None:
        raise ConnectionError("cache unavailable")

    async def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        raise ConnectionError("cache unavailable")

    async def close(self) -> None:
        pass


@pytest.fixture
def raising_cache():
    return RaisingCache()


@pytest_asyncio.fixture
async def client_with_raising_cache(db_session, raising_cache):
    """Same as `client`, but the cache backend always raises -- used to
    exercise AttendanceService's cache-outage fallback to the DB unique
    constraint.
    """
    from src.main import app

    async def _get_db():
        yield db_session

    async def _get_redis():
        return raising_cache

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_redis] = _get_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(db_session, fake_cache):
    from src.main import app

    async def _get_db():
        yield db_session

    async def _get_redis():
        return fake_cache

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_redis] = _get_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_embedding():
    """A deterministic, normalized 512-dim vector for use as a face embedding."""
    vec = [0.0] * 512
    vec[0] = 1.0
    return vec


@pytest.fixture
def sample_embedding_far():
    """A different deterministic 512-dim vector, orthogonal to sample_embedding."""
    vec = [0.0] * 512
    vec[1] = 1.0
    return vec
