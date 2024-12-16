from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from ..config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate the API key from the X-API-Key header.
    Returns the API key if valid, raises HTTPException if not.
    """
    if api_key == settings.API_SECRET_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
