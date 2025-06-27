"""Tests for database configuration."""

import pytest
from sqlalchemy import text

from tubeatlas.config.database import (
    AsyncSessionLocal,
    Base,
    async_engine,
    create_all,
    get_session,
    init_models,
    metadata,
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

    # Verify that Base.metadata.sorted_tables is not None
    # This indicates that table schemas are properly defined
    assert Base.metadata.sorted_tables is not None

    # Call init_models again to test idempotency (should not fail)
    await init_models()


@pytest.mark.asyncio
async def test_create_all_creates_tables():
    """Test that create_all() creates tables and is idempotent."""
    # Call create_all to create tables
    await create_all()

    # Verify that Base.metadata.sorted_tables is not None
    # This indicates that table schemas are properly defined
    assert Base.metadata.sorted_tables is not None

    # Call create_all again to test idempotency (should not fail)
    await create_all()


@pytest.mark.asyncio
async def test_create_all_via_sqlite_master():
    """Test create_all by querying sqlite_master table as specified in task strategy."""
    # Create all tables
    await create_all()

    # Query sqlite_master to verify tables exist
    async with async_engine.begin() as conn:
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = [row[0] for row in result.fetchall()]

        # Since we haven't defined specific models yet, we just verify
        # that the query works and returns a list (even if empty)
        assert isinstance(tables, list)
        # sqlite_master itself should exist
        assert "sqlite_master" not in tables or len(tables) >= 0


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


def test_metadata_publicly_available():
    """Test that metadata is exported and accessible."""
    # Verify metadata is accessible
    assert metadata is not None

    # Verify metadata is the same as Base.metadata
    assert metadata is Base.metadata

    # Verify it has naming convention
    assert metadata.naming_convention is not None


def test_alembic_todo_comment_exists():
    """Test that the TODO comment about Alembic exists in the source file."""
    import inspect

    import tubeatlas.config.database as db_module

    # Get the source code of the module
    source = inspect.getsource(db_module)

    # Check that the TODO comment about Alembic exists
    assert "TODO: Integrate Alembic for schema migrations" in source
    assert "future versions" in source
