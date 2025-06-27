"""Celery app entry point for command line usage."""

from .config.celery_app import celery_app

__all__ = ["celery_app"]
