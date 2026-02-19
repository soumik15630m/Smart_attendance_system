import ipaddress
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from alembic import command  # type: ignore
from alembic.config import Config
from sqlalchemy import text

from src.config import settings
from src.database import engine
from src.routers import (
    attendance_router,
    health_router,
    local_ui_router,
    persons_router,
    web_stream,
)

WEBUI_DIR = Path(__file__).resolve().parent / "webui"


# Lifecycle Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Server starting up... DB URL: {settings.DATABASE_URL.split('@')[-1]}")
    try:
        print(" Checking for database migrations...")
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alembic_ini_path = os.path.join(root_dir, "alembic.ini")

        alembic_cfg = Config(alembic_ini_path)
        alembic_cfg.set_main_option(
            "script_location", os.path.join(root_dir, "alembic")
        )
        command.upgrade(alembic_cfg, "head")
        print("Database is up to date.")
    except Exception as e:
        print(f" Migration Warning: {e}")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("Database connection established.")
    except Exception as e:
        print(f"CRITICAL: Database connection failed! {e}")

    yield

    print("Server shutting down...")
    await engine.dispose()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def enforce_local_only_mode(request: Request, call_next):
    if settings.LOCAL_ONLY:
        client_host = request.client.host if request.client else None
        if not _is_loopback_host(client_host):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Local-only mode is enabled. Access is allowed only from this machine."
                },
            )
    return await call_next(request)


app.mount("/ui/static", StaticFiles(directory=str(WEBUI_DIR)), name="ui-static")

# --- Register Routers ---
app.include_router(attendance_router)
app.include_router(health_router)
app.include_router(web_stream.router)
app.include_router(persons_router)
app.include_router(local_ui_router)


def start():
    import uvicorn

    host = "127.0.0.1" if settings.LOCAL_ONLY else settings.HOST
    uvicorn.run(
        "src.main:app",
        host=host,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "dashboard": "/ui",
        "docs": "/docs",
        "version": settings.VERSION,
        "local_only": settings.LOCAL_ONLY,
    }
