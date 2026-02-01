from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Attendance System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database (Neon PostgreSQL)
    # format: postgresql+asyncpg://user:password@host:port/dbname
    DATABASE_URL: str

    # Redis (Upstash or Local)
    # format: redis://host:port/0
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "Hi Soumik here....."

    # Face Recognition Thresholds
    # 0.6 is strict, 0.4 is loose. 0.5 is a good start for InsightFace.
    SIMILARITY_THRESHOLD: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()