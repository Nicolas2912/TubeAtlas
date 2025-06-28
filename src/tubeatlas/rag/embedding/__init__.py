"""
Embedding package for converting text to vector representations.
"""

from .base import EmbedderInterface
from .openai import OpenAIEmbedder

__all__ = [
    "EmbedderInterface",
    "OpenAIEmbedder",
]
