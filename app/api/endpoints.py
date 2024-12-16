from fastapi import APIRouter, Depends, HTTPException, Path, Body
from sqlalchemy.orm import Session
from ..database.database import get_db
import numpy as np
from ..database.models import Document, DocumentChunk
from ..embedding.embedder import PlaceholderEmbedder
from ..config import settings
from ..ingestion.downloader import GutenbergDownloader
from ..ingestion.preprocessor import TextPreprocessor
from . import auth
import logging
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from typing import Dict
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter()

request_counts: Dict[str, Dict] = {}


def rate_limit(api_key: str = Depends(auth.get_api_key)):
    now = datetime.now()
    if api_key not in request_counts:
        request_counts[api_key] = {"count": 1, "reset_time": now + timedelta(minutes=1)}
    else:
        if now > request_counts[api_key]["reset_time"]:
            request_counts[api_key] = {
                "count": 1,
                "reset_time": now + timedelta(minutes=1),
            }
        else:
            request_counts[api_key]["count"] += 1
            if request_counts[api_key]["count"] > 100:  # 100 requests per minute
                raise HTTPException(status_code=429, detail="Too many requests")


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=1000, description="Search query text"
    )
    top_n: int = Field(5, ge=1, le=100, description="Number of results to return")

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


@router.post("/search/")
async def search_documents(
    search_request: SearchRequest,
    api_key: str = Depends(rate_limit),
    db: Session = Depends(get_db),
):
    """
    Search for documents using semantic similarity.
    """
    try:
        embedder = PlaceholderEmbedder()
        query_embedding = embedder.embed_text(search_request.query)

        chunks = db.query(DocumentChunk).all()

        similarities = []
        for chunk in chunks:
            similarity = float(
                np.dot(query_embedding, chunk.embedding)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding))
            )
            similarities.append((similarity, chunk))

        top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[
            : search_request.top_n
        ]

        results = []
        for similarity, chunk in top_chunks:
            doc = db.query(Document).filter(Document.id == chunk.document_id).first()
            results.append(
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "chunk_content": chunk.content,
                    "similarity": float(similarity),
                }
            )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@router.post("/ingest/{book_id}")
async def ingest_book(
    book_id: int = Path(..., description="Project Gutenberg book ID", ge=1, le=1000000),
    api_key: str = Depends(auth.get_api_key),
    db: Session = Depends(get_db),
):
    try:
        logger.info(f"Ingestion started for book_id {book_id}")
        downloader = GutenbergDownloader()
        preprocessor = TextPreprocessor(chunk_size=settings.CHUNK_SIZE)
        embedder = PlaceholderEmbedder(embedding_dim=settings.EMBEDDING_DIM)

        book_content = downloader.download_book(book_id)
        if book_content.startswith("Failed"):
            raise HTTPException(status_code=404, detail=f"Book {book_id} not found")

        if len(book_content) > 10_000_000:  # 10MB limit
            raise HTTPException(status_code=413, detail="Content too large")

        document = Document(
            title=f"Book {book_id}",
            author="Unknown",
            content=book_content,
            is_privileged=False,
        )
        db.add(document)
        db.flush()

        chunks = preprocessor.split_into_chunks(book_content)
        for chunk_text in chunks:
            embedding = embedder.embed_text(chunk_text)
            embedding_list = [float(x) for x in embedding]
            chunk = DocumentChunk(
                document_id=document.id, content=chunk_text, embedding=embedding_list
            )
            db.add(chunk)

        db.commit()

        return {
            "status": "success",
            "document_id": document.id,
            "chunks_processed": len(chunks),
        }

    except Exception as e:
        logger.error(f"Error during ingestion of book {book_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        db.close()
