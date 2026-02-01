from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.database import engine
# Update the import to include persons_router
from src.routers import attendance_router, health_router, web_stream, persons_router

# Lifecycle Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Server starting up... DB URL: {settings.DATABASE_URL.split('@')[-1]}")
    yield
    print("Server shutting down...")
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

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
app.include_router(persons_router)  # <--- Add this line

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "version": settings.VERSION
    }