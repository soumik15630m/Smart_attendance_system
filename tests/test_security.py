import pytest
from fastapi import HTTPException

from src.config import settings
from src.utils.security import verify_api_key


@pytest.mark.asyncio
async def test_no_check_when_api_key_unset(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    await verify_api_key(x_api_key=None)  # should not raise


@pytest.mark.asyncio
async def test_rejects_missing_header_when_api_key_set(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(x_api_key="wrong")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret123")
    await verify_api_key(x_api_key="secret123")  # should not raise
