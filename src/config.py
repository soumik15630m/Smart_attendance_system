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

    # 0.5 is a good default for InsightFace.
    SIMILARITY_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
