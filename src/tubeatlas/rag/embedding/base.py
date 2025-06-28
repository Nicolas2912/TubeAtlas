# flake8: noqa
# mypy: ignore-errors

"""
Base interface for text embedding models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class EmbedderInterface(ABC):
    """
    Abstract base class for text embedders.

    All embedders must implement the embed_texts method to convert
    text strings into vector representations.
    """

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> List[List[float]]:
        """
        Convert a list of texts into embeddings.

        Args:
            texts: List of text strings to embed
            model: Optional model name override
            batch_size: Number of texts to process in each batch
            **kwargs: Additional model-specific parameters

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            ValueError: If texts is empty or contains invalid content
            RuntimeError: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_text(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> List[float]:
        """
        Convert a single text into an embedding.

        Args:
            text: Text string to embed
            model: Optional model name override
            **kwargs: Additional model-specific parameters

        Returns:
            Embedding vector as a list of floats

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get the dimension of embeddings produced by this embedder.

        Args:
            model: Optional model name

        Returns:
            Embedding dimension (number of features)
        """
        pass

    @abstractmethod
    def get_max_input_length(self, model: Optional[str] = None) -> int:
        """
        Get the maximum input length supported by this embedder.

        Args:
            model: Optional model name

        Returns:
            Maximum input length in tokens
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters for this embedder.

        Returns:
            Dictionary of configuration parameters
        """
        pass

    def validate_texts(self, texts: List[str]) -> None:
        """
        Validate input texts before embedding.

        Args:
            texts: List of texts to validate

        Raises:
            ValueError: If texts are invalid
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        if not isinstance(texts, list):
            raise ValueError("texts must be a list")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"text at index {i} must be a string")
            if not text.strip():
                raise ValueError(
                    f"text at index {i} cannot be empty or whitespace only"
                )

    def validate_text(self, text: str) -> None:
        """
        Validate a single input text before embedding.

        Args:
            text: Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace only")

    def chunk_long_text(
        self, text: str, max_length: int, overlap: int = 0
    ) -> List[str]:
        """
        Split long text into chunks that fit within max_length.

        Args:
            text: Text to chunk
            max_length: Maximum length per chunk (in characters)
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_length, len(text))
            chunk = text[start:end]

            # Try to break at word boundaries
            if end < len(text) and not text[end].isspace():
                # Find the last space in the chunk
                last_space = chunk.rfind(" ")
                if (
                    last_space > start + max_length // 2
                ):  # Only if it's not too far back
                    end = start + last_space
                    chunk = text[start:end]

            chunks.append(chunk.strip())

            # Move start position with overlap consideration
            start = max(start + 1, end - overlap)

        return [chunk for chunk in chunks if chunk.strip()]
