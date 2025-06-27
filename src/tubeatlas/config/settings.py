"""Application settings configuration."""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    youtube_api_key: Optional[str] = None

    # Database Configuration
    database_url: str = "sqlite+aiosqlite:///tubeatlas.db"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Application Configuration
    app_name: str = "TubeAtlas"
    app_version: str = "2.0.0"
    debug: bool = False

    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    api_workers: int = 4

    # Token Limits
    max_tokens_per_request: int = 100000
    max_tokens_per_video: int = 1000000

    # Processing Configuration
    default_language: str = "en"
    max_videos_per_channel: int = 1000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignore extra environment variables
    }


# Global settings instance
settings = Settings()
