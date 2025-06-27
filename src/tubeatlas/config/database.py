"""Database configuration."""

from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .settings import settings

# Define naming convention for constraints to avoid Alembic conflicts
naming_convention = {
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "ix": "ix_%(table_name)s_%(column_0_name)s",
}

# Create metadata with naming convention
metadata = MetaData(naming_convention=naming_convention)

# Base class for all models
Base = declarative_base(metadata=metadata)

# Create async engine with pool_size as specified
async_engine = create_async_engine(
    settings.database_url, pool_size=20, echo=False, future=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# TODO: Integrate Alembic for schema migrations in future versions
# This function provides basic table creation for development and testing
async def create_all() -> None:
    """Create all database tables.

    This function creates all tables defined by the Base metadata.
    It is idempotent and safe to call during application startup
    or test setup. It will not fail if tables already exist.

    Note: This is a simple table creation utility. For production
    applications, consider using Alembic for proper schema migrations.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def init_models() -> None:
    """Initialize database models (create all tables).

    This function is idempotent and safe to call during application startup
    or test setup. It will not fail if tables already exist.

    Alias for create_all() for backward compatibility.
    """
    await create_all()


# Export commonly used components
__all__ = [
    "async_engine",
    "AsyncSessionLocal",
    "Base",
    "metadata",
    "get_session",
    "create_all",
    "init_models",
]
