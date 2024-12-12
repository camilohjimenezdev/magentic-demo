from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from ..config import settings

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key