from __future__ import annotations

import logging
from typing import Type

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from ..utils.exceptions import (
    ConfigurationException,
    DatabaseException,
    KnowledgeGraphGenerationException,
    QuotaExceededError,
    TokenLimitExceededException,
    TranscriptDownloadError,
    TransientAPIError,
    TubeAtlasException,
    ValidationException,
    YouTubeAPIException,
)

logger = logging.getLogger(__name__)


_DEFAULT_STATUS_MAP: dict[Type[TubeAtlasException], int] = {
    ValidationException: 422,
    QuotaExceededError: 429,
    TransientAPIError: 502,
    YouTubeAPIException: 502,
    DatabaseException: 503,
    KnowledgeGraphGenerationException: 500,
    TokenLimitExceededException: 400,
    TranscriptDownloadError: 502,
    ConfigurationException: 500,
}


def _status_code_for(exc: TubeAtlasException) -> int:
    """Return an appropriate HTTP status code for a TubeAtlasException subclass."""
    for exc_type, code in _DEFAULT_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            return code
    return 500  # generic fallback


def register_exception_handlers(app) -> None:  # type: ignore[valid-type]
    """Attach global exception handlers to the provided FastAPI app."""

    @app.exception_handler(TubeAtlasException)  # type: ignore[arg-type]
    async def _handle_tubeatlas_exceptions(
        request: Request, exc: TubeAtlasException
    ):  # noqa: WPS430
        status_code = _status_code_for(exc)
        logger.error(
            "%s on %s %s → %d: %s",
            exc.__class__.__name__,
            request.method,
            request.url.path,
            status_code,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
        )

    @app.exception_handler(SQLAlchemyError)  # type: ignore[arg-type]
    async def _handle_sqlalchemy_error(
        request: Request, exc: SQLAlchemyError
    ):  # noqa: WPS430
        logger.error(
            "Database error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": "database_error",
                "message": "Database operation failed. Please try again later.",
            },
        )
