import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from alembic.config import Config
from alembic import command  # type: ignore

from src.config import settings
from src.database import engine
from src.routers import attendance_router, health_router, web_stream, persons_router


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

# --- Register Routers ---
app.include_router(attendance_router)
app.include_router(health_router)
app.include_router(web_stream.router)
app.include_router(persons_router)


def start():
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "version": settings.VERSION,
    }
