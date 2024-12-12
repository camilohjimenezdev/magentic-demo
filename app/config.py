# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL connection string format:
    # postgresql://user:password@host:port/database_name
    DATABASE_URL: str = "postgresql://postgres:beckham23@localhost:5432/semantic_search"
    API_SECRET_KEY: str = "your-secret-key"
    CHUNK_SIZE: int = 1000
    EMBEDDING_DIM: int = 384

    class Config:
        env_file = ".env"


settings = Settings()
