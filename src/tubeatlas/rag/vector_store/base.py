"""
Base interface for vector stores.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..chunking.base import Chunk


class VectorStoreInterface(ABC):
    """
    Abstract base class for vector stores.

    Vector stores handle storage and retrieval of embeddings with associated metadata.
    """

    @abstractmethod
    def build_index(
        self, chunks: List[Chunk], embeddings: List[List[float]], **kwargs
    ) -> None:
        """
        Build the vector index from chunks and their embeddings.

        Args:
            chunks: List of Chunk objects containing text and metadata
            embeddings: List of embedding vectors corresponding to chunks
            **kwargs: Additional build parameters

        Raises:
            ValueError: If chunks and embeddings don't match in length
            RuntimeError: If index building fails
        """
        pass

    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: List[List[float]], **kwargs) -> None:
        """
        Add new chunks and embeddings to existing index.

        Args:
            chunks: List of new Chunk objects
            embeddings: List of embedding vectors for new chunks
            **kwargs: Additional parameters

        Raises:
            ValueError: If chunks and embeddings don't match in length
            RuntimeError: If adding fails
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query vector to search with
            k: Number of results to return
            filters: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            List of (Chunk, similarity_score) tuples, sorted by similarity

        Raises:
            ValueError: If query_embedding has wrong dimensions
            RuntimeError: If search fails
        """
        pass

    @abstractmethod
    def get_by_ids(self, chunk_ids: List[str]) -> List[Optional[Chunk]]:
        """
        Retrieve chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of Chunk objects (None for missing IDs)
        """
        pass

    @abstractmethod
    def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks actually deleted
        """
        pass

    @abstractmethod
    def persist(self, path: Union[str, Path]) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the store

        Raises:
            IOError: If saving fails
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the store from

        Raises:
            IOError: If loading fails
            FileNotFoundError: If path doesn't exist
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats like count, dimension, memory usage, etc.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all data from the vector store.
        """
        pass

    def validate_embeddings(
        self, embeddings: List[List[float]], expected_dim: Optional[int] = None
    ) -> None:
        """
        Validate embedding vectors.

        Args:
            embeddings: List of embedding vectors to validate
            expected_dim: Expected dimension (inferred if None)

        Raises:
            ValueError: If embeddings are invalid
        """
        self._validate_embeddings_structure(embeddings)
        first_embedding = embeddings[0]
        expected_dim = self._determine_expected_dimension(first_embedding, expected_dim)
        self._validate_all_embeddings(embeddings, expected_dim)

    def _validate_embeddings_structure(self, embeddings: List[List[float]]) -> None:
        """Validate basic structure of embeddings."""
        if not embeddings:
            raise ValueError("embeddings cannot be empty")

        if not isinstance(embeddings, list):
            raise ValueError("embeddings must be a list")

        first_embedding = embeddings[0]
        if not isinstance(first_embedding, list):
            raise ValueError("each embedding must be a list of floats")

        if not first_embedding:
            raise ValueError("embeddings cannot be empty vectors")

    def _determine_expected_dimension(
        self, first_embedding: List[float], expected_dim: Optional[int]
    ) -> int:
        """Determine and validate expected dimension."""
        if expected_dim is None:
            return len(first_embedding)
        elif len(first_embedding) != expected_dim:
            raise ValueError(
                f"first embedding has dimension {len(first_embedding)}, "
                f"expected {expected_dim}"
            )
        return expected_dim

    def _validate_all_embeddings(
        self, embeddings: List[List[float]], expected_dim: int
    ) -> None:
        """Validate all embeddings have correct dimension and values."""
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, list):
                raise ValueError(f"embedding at index {i} must be a list")

            if len(embedding) != expected_dim:
                raise ValueError(
                    f"embedding at index {i} has dimension {len(embedding)}, "
                    f"expected {expected_dim}"
                )

            self._validate_embedding_values(embedding, i)

    def _validate_embedding_values(self, embedding: List[float], index: int) -> None:
        """Validate individual embedding values are numeric."""
        for j, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"embedding at index {index}, position {j} "
                    f"must be numeric, got {type(value)}"
                )

    def validate_chunks_and_embeddings(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        """
        Validate that chunks and embeddings are compatible.

        Args:
            chunks: List of chunks
            embeddings: List of embeddings

        Raises:
            ValueError: If validation fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if not chunks:
            raise ValueError("chunks cannot be empty")

        # Validate chunks
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"item at index {i} must be a Chunk object")
            if not chunk.id:
                raise ValueError(f"chunk at index {i} must have a non-empty id")

        # Validate embeddings
        self.validate_embeddings(embeddings)
