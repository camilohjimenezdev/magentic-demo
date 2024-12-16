"""
Test configuration file that defines shared pytest fixtures for all test files.
These fixtures provide common test setup and teardown functionality.
"""

import pytest
from app.config import settings
from app.database.database import engine, Base, get_db
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """
    Creates and returns a FastAPI TestClient instance.

    Returns:
        TestClient: A test client that can be used to make HTTP requests to the FastAPI application
    """
    return TestClient(app)


@pytest.fixture
def test_db():
    """
    Sets up and tears down a test database for each test that uses this fixture.

    Yields:
        None: This fixture doesn't return a value but manages database lifecycle

    Actions:
        1. Creates all database tables before each test
        2. Allows test to run
        3. Drops all tables after test completion
    """
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after tests
    Base.metadata.drop_all(bind=engine)
