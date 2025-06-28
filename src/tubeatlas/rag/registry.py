"""
RAG Component Registry

Provides centralized registration and discovery of pluggable RAG components
including chunkers, embedders, and vector stores.
"""

import logging
from abc import ABC
from typing import Any, Callable, Dict, Optional, Type

from .chunking.base import ChunkerInterface
from .embedding.base import EmbedderInterface
from .vector_store.base import VectorStoreInterface

logger = logging.getLogger(__name__)

# mypy: ignore-errors

# flake8: noqa


class RAGRegistry:
    """
    Singleton registry for RAG components.

    Provides centralized registration and factory methods for:
    - Chunkers (text splitting strategies)
    - Embedders (text to vector conversion)
    - Vector Stores (similarity search backends)
    """

    _instance: Optional["RAGRegistry"] = None

    def __new__(cls) -> "RAGRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._chunkers: Dict[str, Type[ChunkerInterface]] = {}
        self._embedders: Dict[str, Type[EmbedderInterface]] = {}
        self._vector_stores: Dict[str, Type[VectorStoreInterface]] = {}

        # Register default implementations
        self._register_defaults()
        self._initialized = True

        logger.info("Initialized RAG Registry")

    def _register_defaults(self):
        """Register default implementations."""
        try:
            from .chunking.fixed import FixedLengthChunker
            from .chunking.semantic import SemanticChunker
            from .embedding.openai import OpenAIEmbedder
            from .vector_store.faiss_store import FaissVectorStore

            self.register_chunker("fixed", FixedLengthChunker)
            self.register_chunker("semantic", SemanticChunker)
            self.register_embedder("openai", OpenAIEmbedder)
            self.register_vector_store("faiss", FaissVectorStore)

            logger.info("Registered default RAG components")
        except ImportError as e:
            logger.warning(f"Could not register some default components: {e}")

    def register_chunker(
        self, name: str, chunker_class: Type[ChunkerInterface]
    ) -> None:
        """
        Register a chunker implementation.

        Args:
            name: Unique name for the chunker
            chunker_class: Class implementing ChunkerInterface

        Raises:
            ValueError: If name already exists or class doesn't implement interface
        """
        self._validate_registration(
            name, chunker_class, ChunkerInterface, self._chunkers
        )
        self._chunkers[name] = chunker_class
        logger.info(f"Registered chunker: {name} -> {chunker_class.__name__}")

    def register_embedder(
        self, name: str, embedder_class: Type[EmbedderInterface]
    ) -> None:
        """
        Register an embedder implementation.

        Args:
            name: Unique name for the embedder
            embedder_class: Class implementing EmbedderInterface

        Raises:
            ValueError: If name already exists or class doesn't implement interface
        """
        self._validate_registration(
            name, embedder_class, EmbedderInterface, self._embedders
        )
        self._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name} -> {embedder_class.__name__}")

    def register_vector_store(
        self, name: str, store_class: Type[VectorStoreInterface]
    ) -> None:
        """
        Register a vector store implementation.

        Args:
            name: Unique name for the vector store
            store_class: Class implementing VectorStoreInterface

        Raises:
            ValueError: If name already exists or class doesn't implement interface
        """
        self._validate_registration(
            name, store_class, VectorStoreInterface, self._vector_stores
        )
        self._vector_stores[name] = store_class
        logger.info(f"Registered vector store: {name} -> {store_class.__name__}")

    def _validate_registration(
        self,
        name: str,
        component_class: Type,
        interface_class: Type,
        registry: Dict[str, Type],
    ) -> None:
        """Validate component registration."""
        if name in registry:
            raise ValueError(f"Component '{name}' already registered")

        if not issubclass(component_class, interface_class):
            raise ValueError(
                f"Class {component_class.__name__} must implement {interface_class.__name__}"
            )

    def create_chunker(self, name: str, **kwargs) -> ChunkerInterface:
        """
        Create a chunker instance by name.

        Args:
            name: Registered chunker name
            **kwargs: Initialization parameters

        Returns:
            Chunker instance

        Raises:
            ValueError: If chunker name not found
        """
        if name not in self._chunkers:
            raise ValueError(
                f"Unknown chunker: {name}. Available: {list(self._chunkers.keys())}"
            )

        chunker_class = self._chunkers[name]
        return chunker_class(**kwargs)

    def create_embedder(self, name: str, **kwargs) -> EmbedderInterface:
        """
        Create an embedder instance by name.

        Args:
            name: Registered embedder name
            **kwargs: Initialization parameters

        Returns:
            Embedder instance

        Raises:
            ValueError: If embedder name not found
        """
        if name not in self._embedders:
            raise ValueError(
                f"Unknown embedder: {name}. Available: {list(self._embedders.keys())}"
            )

        embedder_class = self._embedders[name]
        return embedder_class(**kwargs)

    def create_vector_store(self, name: str, **kwargs) -> VectorStoreInterface:
        """
        Create a vector store instance by name.

        Args:
            name: Registered vector store name
            **kwargs: Initialization parameters

        Returns:
            Vector store instance

        Raises:
            ValueError: If vector store name not found
        """
        if name not in self._vector_stores:
            raise ValueError(
                f"Unknown vector store: {name}. Available: {list(self._vector_stores.keys())}"
            )

        store_class = self._vector_stores[name]
        return store_class(**kwargs)

    def list_chunkers(self) -> Dict[str, Type[ChunkerInterface]]:
        """Get all registered chunkers."""
        return self._chunkers.copy()

    def list_embedders(self) -> Dict[str, Type[EmbedderInterface]]:
        """Get all registered embedders."""
        return self._embedders.copy()

    def list_vector_stores(self) -> Dict[str, Type[VectorStoreInterface]]:
        """Get all registered vector stores."""
        return self._vector_stores.copy()

    def get_component_info(self, component_type: str) -> Dict[str, Any]:
        """
        Get information about registered components.

        Args:
            component_type: "chunkers", "embedders", or "vector_stores"

        Returns:
            Dictionary with component names and class information
        """
        registries = {
            "chunkers": self._chunkers,
            "embedders": self._embedders,
            "vector_stores": self._vector_stores,
        }

        if component_type not in registries:
            raise ValueError(f"Unknown component type: {component_type}")

        registry = registries[component_type]
        return {
            name: {
                "class_name": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "No documentation available",
            }
            for name, cls in registry.items()
        }

    def clear_registry(self, component_type: Optional[str] = None) -> None:
        """
        Clear registry (mainly for testing).

        Args:
            component_type: Specific type to clear, or None for all
        """
        if component_type is None:
            self._chunkers.clear()
            self._embedders.clear()
            self._vector_stores.clear()
            self._register_defaults()
        elif component_type == "chunkers":
            self._chunkers.clear()
        elif component_type == "embedders":
            self._embedders.clear()
        elif component_type == "vector_stores":
            self._vector_stores.clear()
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        logger.info(f"Cleared registry: {component_type or 'all'}")


# Convenience function to get the singleton instance
def get_registry() -> RAGRegistry:
    """Get the singleton RAG registry instance."""
    return RAGRegistry()
