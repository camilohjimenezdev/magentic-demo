"""
Test suite for API endpoints.

This module contains comprehensive integration tests for the FastAPI endpoints,
testing both the search and ingestion functionality with proper authentication.

Key Test Areas:
1. Authentication validation for all endpoints
2. Input parameter validation
3. Rate limiting functionality
4. Complete workflow testing (ingestion + search)

Test Structure:
- Fixtures for common test setup
- Individual test functions for specific functionality
- Parametrized tests for multiple test cases
- Integration test for end-to-end workflow
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def api_headers():
    """
    Fixture providing headers with API key for authenticated requests.

    Returns:
        dict: Headers dictionary containing the API key
    """
    return {"X-API-Key": settings.API_SECRET_KEY}


def test_search_endpoint_authentication(test_client: TestClient):
    """
    Test that search endpoint requires valid authentication.

    Tests:
        1. Request without API key should fail
        2. Request with invalid API key should fail

    Args:
        test_client: FastAPI test client instance
    """
    # Test without API key
    response = test_client.post("/search/", json={"query": "test", "top_n": 5})
    assert response.status_code in [401, 403]  # Both are acceptable for missing auth

    # Test with invalid API key
    response = test_client.post(
        "/search/",
        json={"query": "test", "top_n": 5},
        headers={"X-API-Key": "invalid-key"},
    )
    assert response.status_code == 401  # Invalid key should be 401


def test_search_endpoint_validation(test_client: TestClient, api_headers):
    """
    Test input validation for search endpoint.

    Validates:
        - Required query parameter
        - top_n parameter bounds (1-100)
        - Query length restrictions

    Args:
        test_client: FastAPI test client instance
        api_headers: Fixture providing valid API headers
    """
    # Test with missing query
    response = test_client.post("/search/", json={"top_n": 5}, headers=api_headers)
    assert response.status_code == 422  # Pydantic validation error

    # Test with invalid top_n (too small)
    response = test_client.post(
        "/search/", json={"query": "test", "top_n": 0}, headers=api_headers
    )
    assert response.status_code == 422  # Pydantic validation error (ge=1)

    # Test with invalid top_n (too large)
    response = test_client.post(
        "/search/", json={"query": "test", "top_n": 101}, headers=api_headers
    )
    assert response.status_code == 422  # Pydantic validation error (le=100)

    # Test with too long query
    long_query = "test" * 1000
    response = test_client.post(
        "/search/", json={"query": long_query, "top_n": 5}, headers=api_headers
    )
    assert response.status_code == 422  # Pydantic validation error (max_length)


def test_ingest_endpoint_authentication(test_client: TestClient):
    """
    Test that ingest endpoint requires valid authentication.

    Tests:
        1. Request without API key should fail
        2. Request with invalid API key should fail

    Args:
        test_client: FastAPI test client instance
    """
    # Test without API key
    response = test_client.post("/ingest/2701")
    assert response.status_code in [401, 403]  # Both are acceptable for missing auth

    # Test with invalid API key
    response = test_client.post(
        "/ingest/2701",
        headers={"X-API-Key": "invalid-key"},
    )
    assert response.status_code == 401  # Invalid key should be 401


def test_ingest_endpoint_validation(test_client: TestClient, api_headers):
    """
    Test input validation for ingest endpoint.

    Validates:
        - Book ID must be positive
        - Book ID must be within valid range

    Args:
        test_client: FastAPI test client instance
        api_headers: Fixture providing valid API headers
    """
    # Test with invalid book ID (negative)
    response = test_client.post("/ingest/-1", headers=api_headers)
    assert response.status_code == 422

    # Test with invalid book ID (too large)
    response = test_client.post("/ingest/9999999", headers=api_headers)
    assert response.status_code == 422


@pytest.fixture
def mock_search_db(monkeypatch):
    """
    Mock database operations for search parameter testing.

    This fixture temporarily replaces the search endpoint's database operations
    with a mock that returns empty results, allowing for validation testing
    without database dependencies.

    Args:
        monkeypatch: pytest's monkeypatch fixture for temporary modifications
    """
    from app.api.endpoints import router

    original_search = router.routes[0].endpoint

    async def mock_search_documents(*args, **kwargs):
        # Return empty results for validation testing
        return []

    monkeypatch.setattr(router.routes[0], "endpoint", mock_search_documents)
    yield
    monkeypatch.setattr(router.routes[0], "endpoint", original_search)


@pytest.mark.parametrize(
    "query,top_n,expected_status",
    [
        ("test query", 5, 200),
        ("", 5, 422),  # Empty query - Pydantic validation
        ("test", 0, 422),  # Invalid top_n - Pydantic validation (ge=1)
        ("test", 101, 422),  # top_n too large - Pydantic validation (le=100)
        ("a" * 1001, 5, 422),  # Query too long - Pydantic validation (max_length)
    ],
)
def test_search_parameters(
    test_client: TestClient,
    api_headers,
    mock_search_db,
    query,
    top_n,
    expected_status,
):
    """
    Parametrized test for various search parameter combinations.

    Tests multiple combinations of search parameters to ensure proper validation:
        - Valid queries
        - Empty queries
        - Invalid top_n values
        - Oversized queries

    Args:
        test_client: FastAPI test client instance
        api_headers: Fixture providing valid API headers
        mock_search_db: Fixture providing mocked database operations
        query: Test query string
        top_n: Number of results to return
        expected_status: Expected HTTP status code
    """
    response = test_client.post(
        "/search/",
        json={"query": query, "top_n": top_n},
        headers=api_headers,
    )
    assert response.status_code == expected_status


@pytest.fixture
def mock_rate_limit_db(monkeypatch):
    """
    Mock database operations for rate limit testing.

    Provides a simplified mock of database operations to test rate limiting
    without actual database dependencies.

    Args:
        monkeypatch: pytest's monkeypatch fixture for temporary modifications
    """
    from app.api.endpoints import router

    original_search = router.routes[0].endpoint

    async def mock_search_documents(*args, **kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(router.routes[0], "endpoint", mock_search_documents)
    yield
    monkeypatch.setattr(router.routes[0], "endpoint", original_search)


@pytest.fixture
def reset_rate_limit():
    """
    Reset rate limiting between tests.

    Clears the request count tracking to ensure clean state between tests.
    """
    from app.api.endpoints import request_counts

    request_counts.clear()
    yield
    request_counts.clear()


def test_rate_limiting(
    test_client: TestClient, api_headers, mock_rate_limit_db, reset_rate_limit
):
    """
    Test that rate limiting is properly enforced.

    Verifies:
        - Requests within limit succeed
        - Requests exceeding limit are blocked
        - Rate limit counter works correctly

    Args:
        test_client: FastAPI test client instance
        api_headers: Fixture providing valid API headers
        mock_rate_limit_db: Fixture providing mocked database operations
        reset_rate_limit: Fixture to reset rate limit tracking
    """
    # Make 101 requests (exceeding the 100 per minute limit)
    responses = []
    for _ in range(101):
        response = test_client.post(
            "/search/",
            json={"query": "test", "top_n": 5},
            headers=api_headers,
        )
        responses.append(response.status_code)
        if response.status_code == 429:  # Too Many Requests
            break

    # Verify that we got rate limited
    assert 429 in responses, "Rate limiting was not enforced"
    # Verify that earlier requests succeeded
    assert 200 in responses, "Initial requests should succeed"


@pytest.mark.integration
def test_full_workflow(test_client: TestClient, api_headers, test_db, reset_rate_limit):
    """
    Test complete workflow: ingest a book and then search it.

    This integration test verifies the entire system works together:
        1. Creates necessary database tables
        2. Ingests a test book (Moby Dick)
        3. Performs a search query
        4. Validates search results
        5. Cleans up database

    Args:
        test_client: FastAPI test client instance
        api_headers: Fixture providing valid API headers
        test_db: Fixture providing test database
        reset_rate_limit: Fixture to reset rate limit tracking
    """
    from app.database.models import Base
    from app.database.database import engine

    # Ensure tables are created
    Base.metadata.create_all(bind=engine)

    try:
        # First ingest a book
        book_id = 2701  # Moby Dick
        ingest_response = test_client.post(f"/ingest/{book_id}", headers=api_headers)
        assert ingest_response.status_code == 200
        assert "document_id" in ingest_response.json()

        # Then search for content
        search_response = test_client.post(
            "/search/",
            json={"query": "Call me Ishmael", "top_n": 3},
            headers=api_headers,
        )
        assert search_response.status_code == 200

        results = search_response.json()
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result.get("similarity"), float) for result in results)

    finally:
        # Clean up
        Base.metadata.drop_all(bind=engine)
