"""
Base interface for text chunkers.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils.token_counter import TokenCounter


@dataclass
class Chunk:
    """
    Represents a chunk of text with metadata.

    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        start_idx: Starting character index in original text
        end_idx: Ending character index in original text
        token_count: Number of tokens in the chunk
        metadata: Additional metadata dictionary
    """

    id: str
    text: str
    start_idx: int
    end_idx: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        text: str,
        start_idx: int,
        end_idx: int,
        model: str = "gpt-3.5-turbo",
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Chunk":
        """
        Factory method to create a chunk with automatic token counting.

        Args:
            text: The text content
            start_idx: Starting character index
            end_idx: Ending character index
            model: Model for token counting
            chunk_id: Optional custom ID (generates UUID if None)
            metadata: Optional metadata dictionary

        Returns:
            New Chunk instance
        """
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())

        token_count = TokenCounter.count(text, model)

        return cls(
            id=chunk_id,
            text=text,
            start_idx=start_idx,
            end_idx=end_idx,
            token_count=token_count,
            metadata=metadata or {},
        )


class ChunkerInterface(ABC):
    """
    Abstract base class for text chunkers.

    All chunkers must implement the chunk method to split text into
    manageable pieces with metadata.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If text is invalid
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters for this chunker.

        Returns:
            Dictionary of configuration parameters
        """
        pass

    def validate_text(self, text: str) -> None:
        """
        Validate input text before chunking.

        Args:
            text: Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace only")

    def add_metadata_to_chunks(
        self, chunks: List[Chunk], metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Add metadata to all chunks.

        Args:
            chunks: List of chunks to update
            metadata: Metadata to add

        Returns:
            Updated chunks (modifies in place)
        """
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about a list of chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "total_characters": 0,
                "avg_tokens_per_chunk": 0,
                "avg_chars_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "min_chars": 0,
                "max_chars": 0,
            }

        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "total_characters": sum(char_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
        }
