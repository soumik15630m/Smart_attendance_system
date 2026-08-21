from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Attendance System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOCAL_ONLY: bool = True

    # postgresql+asyncpg://user:pass@host:port/db
    DATABASE_URL: str = ""

    # auto -> upstash -> redis (local auto-start)
    CACHE_BACKEND: str = "auto"
    REDIS_URL: str = "redis://localhost:6379/0"
    AUTO_START_LOCAL_REDIS: bool = True
    LOCAL_REDIS_START_CMD: str = ""
    LOCAL_REDIS_START_TIMEOUT_SECONDS: float = 12.0
    PREFER_DOCKER_REDIS: bool = True
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    SECRET_KEY: str = "Hi Soumik here....."

    # Static header key required on protected endpoints (X-API-Key).
    # Left empty by default so local/dev usage keeps working without setup;
    # it MUST be set in any deployment reachable outside localhost.
    API_KEY: str = ""

    # Comma-separated list of allowed CORS origins. Wildcard ("*") is only
    # honored when DEBUG is True, since "*" combined with credentials is
    # rejected by browsers and is unsafe in production anyway.
    ALLOWED_ORIGINS: str = "http://localhost:8000,http://127.0.0.1:8000"

    # 0.5 is a good default for InsightFace.
    SIMILARITY_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        if self.DEBUG and not self.ALLOWED_ORIGINS.strip():
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()

if not settings.DEBUG:
    if not settings.API_KEY:
        raise RuntimeError(
            "API_KEY must be set (via .env or environment) when DEBUG is False."
        )
    if settings.SECRET_KEY == "Hi Soumik here.....":
        raise RuntimeError(
            "SECRET_KEY must be changed from its default value when DEBUG is False."
        )
