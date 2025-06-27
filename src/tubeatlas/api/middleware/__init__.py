"""API middleware."""

from .database import DatabaseHealthMiddleware

__all__ = ["DatabaseHealthMiddleware"]
