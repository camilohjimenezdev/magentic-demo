# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL connection string format:
    # postgresql://user:password@host:port/database_name
    DATABASE_URL: str
    API_SECRET_KEY: str
    CHUNK_SIZE: int = 1000
    EMBEDDING_DIM: int = 384

    class Config:
        env_file = ".env"


settings = Settings()
