"""
Fixed-length chunker implementation with sliding window.
"""

import re
from typing import Any, Dict, List

from ...utils.token_counter import TokenCounter
from .base import Chunk, ChunkerInterface


class FixedLengthChunker(ChunkerInterface):
    """
    Chunker that splits text into fixed-length chunks with overlap.

    Uses token-based length calculation and attempts to preserve
    sentence boundaries when possible.
    """

    def __init__(
        self,
        length_tokens: int = 512,
        overlap_tokens: int = 64,
        model: str = "gpt-3.5-turbo",
        preserve_sentences: bool = True,
        sentence_boundary_tolerance: int = 20,
    ):
        """
        Initialize fixed-length chunker.

        Args:
            length_tokens: Target length of each chunk in tokens
            overlap_tokens: Number of tokens to overlap between chunks
            model: Model name for token counting
            preserve_sentences: Whether to try preserving sentence boundaries
            sentence_boundary_tolerance: Max tokens to adjust for sentence boundaries
        """
        if length_tokens <= 0:
            raise ValueError("length_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if overlap_tokens >= length_tokens:
            raise ValueError("overlap_tokens must be less than length_tokens")

        self.length_tokens = length_tokens
        self.overlap_tokens = overlap_tokens
        self.model = model
        self.preserve_sentences = preserve_sentences
        self.sentence_boundary_tolerance = sentence_boundary_tolerance

        # Sentence boundary regex
        self.sentence_pattern = re.compile(r"[.!?]+\s+")

    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into fixed-length chunks with overlap.

        Args:
            text: Input text to chunk

        Returns:
            List of Chunk objects
        """
        self.validate_text(text)

        if not text.strip():
            return []

        chunks = []
        start_pos = 0
        chunk_id = 0

        while start_pos < len(text):
            # Calculate end position for this chunk
            end_pos = self._find_chunk_end(text, start_pos)

            if end_pos <= start_pos:
                break

            # Extract chunk text
            chunk_text = text[start_pos:end_pos].strip()

            if chunk_text:
                # Create chunk with automatic token counting
                chunk = Chunk.create(
                    text=chunk_text,
                    start_idx=start_pos,
                    end_idx=end_pos,
                    model=self.model,
                    chunk_id=f"chunk_{chunk_id}",
                    metadata={
                        "chunker": "fixed",
                        "length_tokens": self.length_tokens,
                        "overlap_tokens": self.overlap_tokens,
                        "model": self.model,
                    },
                )
                chunks.append(chunk)
                chunk_id += 1

            # Calculate next start position with overlap
            if end_pos >= len(text):
                break

            overlap_chars = self._calculate_overlap_chars(text, start_pos, end_pos)
            start_pos = max(start_pos + 1, end_pos - overlap_chars)

        return chunks

    def _find_chunk_end(self, text: str, start_pos: int) -> int:
        """
        Find the end position for a chunk starting at start_pos.

        Args:
            text: Full text
            start_pos: Starting position

        Returns:
            End position for the chunk
        """
        # Estimate characters per token for initial positioning
        sample_text = text[start_pos : start_pos + min(1000, len(text) - start_pos)]
        if sample_text:
            sample_tokens = TokenCounter.count(sample_text, self.model)
            chars_per_token = len(sample_text) / max(1, sample_tokens)
        else:
            chars_per_token = 4  # Default estimate

        # Initial estimate for end position
        estimated_chars = int(self.length_tokens * chars_per_token)
        initial_end = min(start_pos + estimated_chars, len(text))

        # Binary search to find exact token-based end position
        end_pos = self._binary_search_token_boundary(
            text, start_pos, initial_end, self.length_tokens
        )

        # Try to preserve sentence boundaries if enabled
        if self.preserve_sentences and end_pos < len(text):
            adjusted_end = self._adjust_for_sentence_boundary(text, start_pos, end_pos)
            if adjusted_end != end_pos:
                # Verify the adjustment doesn't exceed token tolerance
                chunk_text = text[start_pos:adjusted_end]
                token_count = TokenCounter.count(chunk_text, self.model)
                if (
                    abs(token_count - self.length_tokens)
                    <= self.sentence_boundary_tolerance
                ):
                    end_pos = adjusted_end

        return end_pos

    def _binary_search_token_boundary(
        self, text: str, start_pos: int, initial_end: int, target_tokens: int
    ) -> int:
        """
        Use binary search to find position that gives target token count.

        Args:
            text: Full text
            start_pos: Starting position
            initial_end: Initial end position estimate
            target_tokens: Target number of tokens

        Returns:
            Position that gives approximately target_tokens
        """
        left = start_pos + 1
        right = min(initial_end + 1000, len(text))  # Add buffer for search
        best_pos = initial_end

        # Limit iterations to prevent infinite loops
        for _ in range(20):
            if left >= right:
                break

            mid = (left + right) // 2
            chunk_text = text[start_pos:mid]
            token_count = TokenCounter.count(chunk_text, self.model)

            if token_count == target_tokens:
                return mid
            elif token_count < target_tokens:
                left = mid + 1
                best_pos = mid
            else:
                right = mid

        return min(best_pos, len(text))

    def _adjust_for_sentence_boundary(
        self, text: str, start_pos: int, end_pos: int
    ) -> int:
        """
        Adjust end position to align with sentence boundaries.

        Args:
            text: Full text
            start_pos: Starting position
            end_pos: Current end position

        Returns:
            Adjusted end position
        """
        # Look for sentence boundaries near the end position
        search_start = max(start_pos, end_pos - 200)
        search_end = min(len(text), end_pos + 200)
        search_text = text[search_start:search_end]

        # Find sentence boundaries
        sentence_ends = []
        for match in self.sentence_pattern.finditer(search_text):
            abs_pos = search_start + match.end()
            if start_pos < abs_pos <= search_end:
                sentence_ends.append(abs_pos)

        if not sentence_ends:
            return end_pos

        # Find the closest sentence boundary to our target end
        closest_end = min(sentence_ends, key=lambda x: abs(x - end_pos))

        # Only adjust if it's within reasonable distance
        if abs(closest_end - end_pos) <= 100:  # Max 100 characters adjustment
            return closest_end

        return end_pos

    def _calculate_overlap_chars(self, text: str, start_pos: int, end_pos: int) -> int:
        """
        Calculate character overlap based on token overlap.

        Args:
            text: Full text
            start_pos: Current chunk start
            end_pos: Current chunk end

        Returns:
            Number of characters for overlap
        """
        if self.overlap_tokens == 0:
            return 0

        chunk_text = text[start_pos:end_pos]
        chunk_tokens = TokenCounter.count(chunk_text, self.model)

        if chunk_tokens <= self.overlap_tokens:
            return len(chunk_text) // 2  # Overlap half the chunk

        # Estimate characters per token for this chunk
        chars_per_token = len(chunk_text) / chunk_tokens
        overlap_chars = int(self.overlap_tokens * chars_per_token)

        return min(overlap_chars, len(chunk_text) // 2)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "type": "fixed",
            "length_tokens": self.length_tokens,
            "overlap_tokens": self.overlap_tokens,
            "model": self.model,
            "preserve_sentences": self.preserve_sentences,
            "sentence_boundary_tolerance": self.sentence_boundary_tolerance,
        }
