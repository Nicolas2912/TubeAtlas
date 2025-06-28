"""
TubeAtlas RAG (Retrieval-Augmented Generation) Package

A comprehensive RAG implementation with modular components for chunking,
embedding, vector storage, and retrieval pipelines.

Core Components:
- TokenCounter: Utility for counting tokens across different models
- Chunking: Text splitting strategies (fixed-length, semantic)
- Embedding: Text embedding generation (OpenAI, custom)
- Vector Store: Vector storage and similarity search (FAISS, custom)
- Pipeline: End-to-end ingestion and retrieval workflows
- Registry: Component factory and management
"""

# Import TokenCounter from utils
from ..utils.token_counter import TokenCounter

# Benchmarking
from .benchmarks.benchmark import RAGBenchmark

# Core interfaces
from .chunking.base import Chunk, ChunkerInterface

# Concrete implementations
from .chunking.fixed import FixedLengthChunker
from .chunking.semantic import SemanticChunker
from .embedding.base import EmbedderInterface
from .embedding.openai import OpenAIEmbedder

# Pipeline components
from .pipeline import IngestPipeline, RetrievalPipeline
from .registry import RAGRegistry, get_registry
from .vector_store.base import VectorStoreInterface
from .vector_store.faiss_store import FaissVectorStore

# Export all public components
__all__ = [
    # Core utilities
    "TokenCounter",
    # Base interfaces
    "ChunkerInterface",
    "Chunk",
    "EmbedderInterface",
    "VectorStoreInterface",
    # Concrete implementations
    "FixedLengthChunker",
    "SemanticChunker",
    "OpenAIEmbedder",
    "FaissVectorStore",
    # Pipeline components
    "IngestPipeline",
    "RetrievalPipeline",
    "RAGRegistry",
    "get_registry",
    # Benchmarking
    "RAGBenchmark",
]

# Version info
__version__ = "0.1.0"
