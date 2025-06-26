"""API dependency injection."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.database import AsyncSessionLocal


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 