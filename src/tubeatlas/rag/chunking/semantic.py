"""
Semantic chunker implementation using sentence embeddings.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ...utils.token_counter import TokenCounter
from ..embedding.base import EmbedderInterface
from .base import Chunk, ChunkerInterface

logger = logging.getLogger(__name__)


class SemanticChunker(ChunkerInterface):
    """
    Chunker that groups sentences by semantic similarity.

    Uses an embedder to compute sentence embeddings and groups
    sentences that are semantically similar into chunks.
    """

    def __init__(
        self,
        embedder: EmbedderInterface,
        similarity_threshold: float = 0.92,
        max_chunk_tokens: int = 1024,
        min_chunk_tokens: int = 50,
        token_model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize semantic chunker.

        Args:
            embedder: An instance of a class that implements EmbedderInterface
            similarity_threshold: Cosine similarity threshold for grouping
            max_chunk_tokens: Maximum tokens per chunk
            min_chunk_tokens: Minimum tokens per chunk
            token_model: Model for token counting
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_chunk_tokens <= min_chunk_tokens:
            raise ValueError("max_chunk_tokens must be greater than min_chunk_tokens")

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.token_model = token_model

        # Sentence splitting pattern
        self.sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include in the chunk

        Returns:
            List of Chunk objects
        """
        self.validate_text(text)

        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # If only one sentence, return it as a single chunk
        if len(sentences) == 1:
            chunk_metadata = {
                "chunker": "semantic",
                "similarity_threshold": self.similarity_threshold,
                "embedder_config": self.embedder.get_config(),
                "sentence_count": 1,
            }
            if metadata:
                chunk_metadata.update(metadata)

            return [
                Chunk.create(
                    text=sentences[0]["text"],
                    start_idx=sentences[0]["start_idx"],
                    end_idx=sentences[0]["end_idx"],
                    model=self.token_model,
                    chunk_id="chunk_0",
                    metadata=chunk_metadata,
                )
            ]

        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences, metadata)

        return chunks

    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sentences with position tracking.

        Args:
            text: Input text

        Returns:
            List of sentence dictionaries with text and position info
        """
        sentences = []
        current_pos = 0

        # Split by sentence boundaries
        sentence_texts = self.sentence_pattern.split(text)

        for sentence_text in sentence_texts:
            sentence_text = sentence_text.strip()
            if not sentence_text:
                continue

            # Find the sentence in the original text
            start_idx = text.find(sentence_text, current_pos)
            if start_idx == -1:
                # Fallback: use current position
                start_idx = current_pos

            end_idx = start_idx + len(sentence_text)

            sentences.append(
                {"text": sentence_text, "start_idx": start_idx, "end_idx": end_idx}
            )

            current_pos = end_idx

        return sentences

    def _group_sentences_into_chunks(
        self, sentences: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Group sentences into chunks based on semantic similarity.

        Args:
            sentences: List of sentence dictionaries
            metadata: Optional metadata to include in the chunk

        Returns:
            List of Chunk objects
        """
        if not sentences:
            return []

        # Get embeddings for all sentences
        sentence_texts = [s["text"] for s in sentences]
        embeddings = self.embedder.embed_texts(sentence_texts)

        # Convert to numpy arrays for easier manipulation
        embeddings = np.array(embeddings)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0:1]  # Keep as 2D array
        chunk_id = 0

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i : i + 1]  # Keep as 2D array

            # Calculate similarity with current chunk centroid
            chunk_centroid = np.mean(current_chunk_embedding, axis=0, keepdims=True)
            similarity = self._cosine_similarity(sentence_embedding, chunk_centroid)[
                0, 0
            ]

            # Check if we should add to current chunk or start new one
            exceeds_token_limit = self._would_exceed_token_limit(
                current_chunk_sentences + [sentence]
            )
            should_start_new_chunk = (
                similarity < self.similarity_threshold or exceeds_token_limit
            )

            logger.debug(
                f"Sentence {i}: Similarity={similarity:.4f}, "
                f"Exceeds Limit={exceeds_token_limit}, "
                f"New Chunk={should_start_new_chunk}"
            )

            if should_start_new_chunk:
                # Finalize current chunk
                meets_min = self._meets_minimum_requirements(current_chunk_sentences)
                logger.debug(f"Finalizing chunk. Meets minimum: {meets_min}")
                if meets_min:
                    chunk = self._create_chunk_from_sentences(
                        current_chunk_sentences, chunk_id, metadata
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_embedding
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_embedding = np.vstack(
                    [current_chunk_embedding, sentence_embedding]
                )

        # Don't forget the last chunk
        meets_min = self._meets_minimum_requirements(current_chunk_sentences)
        logger.debug(f"Finalizing last chunk. Meets minimum: {meets_min}")
        if current_chunk_sentences and meets_min:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, chunk_id, metadata
            )
            chunks.append(chunk)

        return chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two embedding arrays.

        Args:
            a: First embedding array
            b: Second embedding array

        Returns:
            Cosine similarity matrix
        """
        # Normalize the vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

        # Calculate cosine similarity
        return np.dot(a_norm, b_norm.T)

    def _would_exceed_token_limit(self, sentences: List[Dict[str, Any]]) -> bool:
        """
        Check if adding sentences would exceed token limit.

        Args:
            sentences: List of sentence dictionaries

        Returns:
            True if token limit would be exceeded
        """
        combined_text = " ".join(s["text"] for s in sentences)
        token_count = TokenCounter.count(combined_text, self.token_model)
        return token_count > self.max_chunk_tokens

    def _meets_minimum_requirements(self, sentences: List[Dict[str, Any]]) -> bool:
        """
        Check if chunk meets minimum requirements.

        Args:
            sentences: List of sentence dictionaries

        Returns:
            True if chunk meets minimum requirements
        """
        if not sentences:
            return False

        combined_text = " ".join(s["text"] for s in sentences)
        token_count = TokenCounter.count(combined_text, self.token_model)
        return token_count >= self.min_chunk_tokens

    def _create_chunk_from_sentences(
        self,
        sentences: List[Dict[str, Any]],
        chunk_id: int,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Create a Chunk object from a list of sentences.

        Args:
            sentences: List of sentence dictionaries
            chunk_id: Unique identifier for the chunk
            base_metadata: Optional base metadata to include

        Returns:
            Chunk object
        """
        combined_text = " ".join(s["text"] for s in sentences)
        start_idx = sentences[0]["start_idx"]
        end_idx = sentences[-1]["end_idx"]

        chunk_metadata = {
            "chunker": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "embedder_config": self.embedder.get_config(),
            "sentence_count": len(sentences),
            "max_chunk_tokens": self.max_chunk_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
        }
        if base_metadata:
            chunk_metadata.update(base_metadata)

        return Chunk.create(
            text=combined_text,
            start_idx=start_idx,
            end_idx=end_idx,
            model=self.token_model,
            chunk_id=f"chunk_{chunk_id}",
            metadata=chunk_metadata,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "type": "semantic",
            "similarity_threshold": self.similarity_threshold,
            "embedder_config": self.embedder.get_config(),
            "max_chunk_tokens": self.max_chunk_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
            "token_model": self.token_model,
        }
