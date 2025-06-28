"""
Vector store package for storing and querying embeddings.
"""

from .base import VectorStoreInterface
from .faiss_store import FaissVectorStore

__all__ = [
    "VectorStoreInterface",
    "FaissVectorStore",
]
