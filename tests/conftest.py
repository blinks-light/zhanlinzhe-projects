"""Shared pytest fixtures for the API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    """Start the application (including model loading) for one test."""
    with TestClient(app) as test_client:
        yield test_client
