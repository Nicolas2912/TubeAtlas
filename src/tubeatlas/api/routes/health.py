"""Health check endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...config.database import get_session
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Dict containing basic health status and app information
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": "development" if settings.debug else "production",
        "app_name": settings.app_name,
    }


@router.get("/db")
async def database_health_check(
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """
    Database connectivity health check endpoint.

    Tests database connection by executing a simple SELECT 1 query.
    Returns 200 OK if successful, or raises HTTPException if database is unavailable.

    Args:
        session: Database session dependency

    Returns:
        Dict containing database health status

    Raises:
        HTTPException: 503 if database connection fails
    """
    try:
        # Execute a simple query to test database connectivity
        result = await session.execute(text("SELECT 1 as health_check"))
        row = result.fetchone()

        if row and row[0] == 1:
            logger.debug("Database health check successful")
            return {
                "status": "healthy",
                "database": "connected",
                "message": "Database connection is working properly",
            }
        else:
            logger.error("Database health check failed: unexpected result")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "database": "error",
                    "message": "Database returned unexpected result",
                },
            )

    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
                "message": "Database connection failed",
                "error": str(e),
            },
        )
