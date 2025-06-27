"""Tests for health check endpoints and database middleware."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy.exc import DisconnectionError

from src.tubeatlas.api.middleware.database import DatabaseHealthMiddleware


def test_middleware_converts_db_errors():
    """Test that database middleware converts DB errors to 503 responses."""
    # Create a simple test app
    test_app = FastAPI()
    middleware = DatabaseHealthMiddleware(test_app)

    async def test_middleware():
        # Create a more complete mock request scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
        }
        request = Request(scope)

        # Mock call_next that raises a database error
        async def call_next_with_error(request):
            raise DisconnectionError("Database disconnected")

        # Test that middleware converts DB error to HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next_with_error)

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["type"] == "database_error"
        assert "Service temporarily unavailable" in exc_info.value.detail["error"]

    # Run the async test
    asyncio.run(test_middleware())


def test_middleware_passes_through_non_db_errors():
    """Test that middleware passes through non-database errors unchanged."""
    # Create a simple test app
    test_app = FastAPI()
    middleware = DatabaseHealthMiddleware(test_app)

    async def test_middleware():
        # Create a more complete mock request scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
        }
        request = Request(scope)

        # Mock call_next that raises a non-database error
        async def call_next_with_error(request):
            raise ValueError("Some other error")

        # Test that middleware re-raises non-DB errors
        with pytest.raises(ValueError):
            await middleware.dispatch(request, call_next_with_error)

    # Run the async test
    asyncio.run(test_middleware())


def test_middleware_successful_request():
    """Test that middleware passes through successful requests unchanged."""
    # Create a simple test app
    test_app = FastAPI()
    middleware = DatabaseHealthMiddleware(test_app)

    async def test_middleware():
        # Create a more complete mock request scope
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
        }
        request = Request(scope)

        # Mock successful response
        class MockResponse:
            status_code = 200
            content = b'{"test": "success"}'

        async def call_next_success(request):
            return MockResponse()

        # Test that middleware passes through successful responses
        response = await middleware.dispatch(request, call_next_success)
        assert response.status_code == 200

    # Run the async test
    asyncio.run(test_middleware())


def test_application_can_be_imported():
    """Test that the FastAPI application can be imported successfully."""
    from src.tubeatlas.main import app

    assert app is not None
    assert hasattr(app, "title")
    assert app.title == "TubeAtlas"
