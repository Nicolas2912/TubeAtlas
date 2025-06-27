"""Database middleware for health checks and error handling."""

import logging
from typing import Callable

from fastapi import HTTPException, Request, Response
from sqlalchemy.exc import DatabaseError, DisconnectionError, TimeoutError
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class DatabaseHealthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle database connection errors and transform them to 503 responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and handle database-related exceptions.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            Response: HTTP response, potentially modified for database errors
        """
        try:
            response = await call_next(request)
            return response
        except (DatabaseError, DisconnectionError, TimeoutError) as db_error:
            # Log the database error for monitoring
            logger.error(
                f"Database error in {request.method} {request.url.path}: {str(db_error)}",
                exc_info=True,
            )

            # Transform database errors into 503 Service Unavailable
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service temporarily unavailable",
                    "message": "Database connection issue. Please try again later.",
                    "type": "database_error",
                },
            )
        except Exception as e:
            # Re-raise non-database exceptions unchanged
            raise e
