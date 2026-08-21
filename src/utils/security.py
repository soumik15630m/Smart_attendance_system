from fastapi import Header, HTTPException, status

from src.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require a valid X-API-Key header on protected routes.

    If API_KEY is left unset (e.g. local development with DEBUG=True), the
    check is skipped so the app keeps working out of the box. In any
    non-debug deployment, config.py enforces that API_KEY is set, so this
    check is always active there.
    """
    if not settings.API_KEY:
        return

    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
