# Semantic Search API

A FastAPI-based semantic search engine that allows ingestion and semantic search of text documents using vector embeddings.

## Features

- 🔍 Semantic search using vector embeddings
- 📚 Document ingestion from Project Gutenberg
- 🔑 API key authentication
- ⚡ Rate limiting (100 requests/minute)
- 📏 Configurable chunk sizes and embedding dimensions

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

## Project Structure

```

app/
├── api/ # API related code
│ ├── auth.py # API key authentication
│ └── endpoints.py # API endpoints
├── database/ # Database related code
│ ├── database.py # Database connection
│ └── models.py # SQLAlchemy models
├── embedding/ # Embedding related code
│ └── embedder.py # Text embedding logic
├── ingestion/ # Document ingestion
├── config.py # Configuration settings
└── main.py # FastAPI application
```

## Database Schema

### Documents Table

- `id`: Primary key
- `title`: Document title
- `author`: Document author
- `content`: Full document content
- `is_privileged`: Access control flag

### Document Chunks Table

- `id`: Primary key
- `document_id`: Foreign key to documents
- `content`: Chunk content
- `embedding`: Vector embedding array

## Features in Detail

### Authentication

- API key-based authentication using `X-API-Key` header
- Configure via `API_SECRET_KEY` environment variable

### Rate Limiting

- 100 requests per minute per API key
- Automatically resets after the minute window

### Vector Search

- Documents are split into chunks during ingestion
- Each chunk is converted to a vector embedding
- Search uses cosine similarity for ranking results

### Document Processing

- Configurable chunk size via `CHUNK_SIZE` setting
- Embeddings dimension controlled by `EMBEDDING_DIM`
- Support for large documents with 10MB size limit

## Development

1. Start PostgreSQL database
2. Set up environment variables
3. Run the application:

```bash
uvicorn app.main:app --reload
```

## Production Deployment

- Uses connection pooling for database efficiencys
- Implements proper error handling and logging
- Includes database connection timeout and retry logic

## Security Considerations

- API key authentication required for all endpoints
- Rate limiting prevents abuse
- Database connection string should be properly secured
- Input validation on all endpoints
