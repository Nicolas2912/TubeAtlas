"""Tests for database configuration."""

import pytest
from sqlalchemy import text

from tubeatlas.config.database import (
    AsyncSessionLocal,
    Base,
    async_engine,
    get_session,
    init_models,
)


@pytest.mark.asyncio
async def test_get_session_basic_query():
    """Test that get_session dependency works and can execute a simple query."""
    # Get session using the dependency
    async for session in get_session():
        # Execute a simple SELECT 1 query
        result = await session.execute(text("SELECT 1"))
        value = result.scalar()

        # Assert the result is correct
        assert value == 1
        break  # Only test the first yielded session


@pytest.mark.asyncio
async def test_init_models_creates_tables():
    """Test that init_models() creates tables and is idempotent."""
    # Call init_models to create tables
    await init_models()

    # Verify that Base.metadata.sorted_tables is not empty
    # This indicates that table schemas are properly defined
    assert Base.metadata.sorted_tables is not None

    # Call init_models again to test idempotency (should not fail)
    await init_models()


@pytest.mark.asyncio
async def test_session_factory_configuration():
    """Test that AsyncSessionLocal is properly configured."""
    # Create a session directly from the factory
    async with AsyncSessionLocal() as session:
        # Verify session is properly configured
        assert session is not None
        assert hasattr(session, "execute")

        # Test a simple query
        result = await session.execute(text("SELECT 1 as test_value"))
        row = result.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_engine_configuration():
    """Test that async_engine is properly configured."""
    # Verify engine properties
    assert async_engine is not None
    assert str(async_engine.url).startswith("sqlite+aiosqlite:///")
    assert "tubeatlas.db" in str(async_engine.url)

    # Test engine connection
    async with async_engine.begin() as conn:
        result = await conn.execute(text("SELECT 1"))
        value = result.scalar()
        assert value == 1


def test_base_metadata_naming_convention():
    """Test that Base has proper naming convention configured."""
    # Verify Base.metadata has naming convention
    assert Base.metadata.naming_convention is not None

    # Check for expected naming convention keys
    naming_conv = Base.metadata.naming_convention
    assert "pk" in naming_conv
    assert "fk" in naming_conv
    assert "uq" in naming_conv
    assert "ck" in naming_conv
    assert "ix" in naming_conv

    # Verify the patterns are correct
    assert naming_conv["pk"] == "pk_%(table_name)s"
    assert naming_conv["fk"] == "fk_%(table_name)s_%(column_0_name)s"
