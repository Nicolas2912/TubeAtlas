"""Text chunking strategies with iterator interfaces and transcript support."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""

    chunk_index: int
    start_pos: int
    end_pos: int
    token_count: Optional[int] = None
    has_timestamps: bool = False
    timestamp_range: Optional[tuple] = None
    source_info: Optional[Dict[str, Any]] = None


@dataclass
class TranscriptChunk:
    """A chunk of transcript text with preserved timestamps and metadata."""

    text: str
    metadata: ChunkMetadata
    timestamps: Optional[List[Dict[str, Any]]] = field(default=None)


def estimate_token_count(text: str) -> int:
    """Estimate token count using character-based approximation."""
    # More accurate approximation: ~4 characters per token for English
    return len(text) // 4


def verify_context_limits(text: str, max_tokens: int) -> bool:
    """Verify that text doesn't exceed model context limits."""
    estimated_tokens = estimate_token_count(text)
    if estimated_tokens > max_tokens:
        logger.warning(
            f"Text exceeds context limit: {estimated_tokens} tokens > {max_tokens} max"
        )
        return False
    return True


def chunk_by_tokens(
    text: str, max_tokens: int = 8000, overlap: int = 200
) -> Generator[TranscriptChunk, None, None]:
    """
    Chunk text by token count with overlap using iterator interface.

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Yields:
        TranscriptChunk objects with text and metadata
    """
    logger.info(f"Chunking text into {max_tokens} token chunks with {overlap} overlap")

    # Character-based approximation
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap * chars_per_token

    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunk_text = text[start:]
        else:
            # Try to find a good breaking point (sentence end)
            break_point = text.rfind(".", start, end)
            if break_point == -1:
                break_point = end
            chunk_text = text[start : break_point + 1]

        # Verify context limits
        if not verify_context_limits(chunk_text, max_tokens):
            logger.error(f"Chunk {chunk_index} exceeds context limits")
            # Try to split further if possible
            if len(chunk_text) > max_chars // 2:
                mid_point = len(chunk_text) // 2
                sentence_break = chunk_text.rfind(".", 0, mid_point)
                if sentence_break > 0:
                    chunk_text = chunk_text[: sentence_break + 1]

        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            start_pos=start,
            end_pos=start + len(chunk_text),
            token_count=estimate_token_count(chunk_text),
            has_timestamps=False,
        )

        yield TranscriptChunk(text=chunk_text, metadata=metadata)

        if end >= len(text):
            break

        chunk_index += 1
        start = start + len(chunk_text) - overlap_chars


def chunk_transcript_with_timestamps(
    transcript_data: str | Dict[str, Any], max_tokens: int = 4000, overlap: int = 100
) -> Generator[TranscriptChunk, None, None]:
    """
    Chunk transcript data while preserving timestamp information.

    Args:
        transcript_data: Transcript text or structured data with timestamps
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Yields:
        TranscriptChunk objects with preserved timestamp data
    """
    logger.info("Chunking transcript data with timestamp preservation")

    # Handle different input formats
    if isinstance(transcript_data, dict):
        # Structured transcript data
        text_content = transcript_data.get("text", "")
        timestamps = transcript_data.get("timestamps", [])
    elif isinstance(transcript_data, str):
        # Raw text - try to extract timestamps
        text_content = transcript_data
        timestamps = _extract_timestamps_from_text(transcript_data)
    else:
        raise ValueError(f"Unsupported transcript data type: {type(transcript_data)}")

    # If no timestamps found, fall back to regular chunking
    if not timestamps:
        logger.info("No timestamps found, falling back to regular chunking")
        for chunk in chunk_by_tokens(text_content, max_tokens, overlap):
            yield chunk
        return

    # Chunk while preserving timestamp boundaries
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token

    current_text = ""
    current_timestamps: List[Dict[str, Any]] = []
    chunk_index = 0
    start_pos = 0

    for i, timestamp_entry in enumerate(timestamps):
        entry_text = timestamp_entry.get("text", "")

        # Check if adding this entry would exceed limits
        test_text = current_text + " " + entry_text if current_text else entry_text

        if len(test_text) > max_chars and current_text:
            # Yield current chunk
            metadata = ChunkMetadata(
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=start_pos + len(current_text),
                token_count=estimate_token_count(current_text),
                has_timestamps=True,
                timestamp_range=(
                    current_timestamps[0].get("start") if current_timestamps else None,
                    current_timestamps[-1].get("end") if current_timestamps else None,
                ),
            )

            chunk = TranscriptChunk(
                text=current_text,
                metadata=metadata,
                timestamps=current_timestamps.copy(),
            )

            # Verify context limits
            if verify_context_limits(current_text, max_tokens):
                yield chunk
            else:
                logger.warning(
                    f"Chunk {chunk_index} exceeds context limits, yielding anyway"
                )
                yield chunk

            # Start new chunk with overlap
            chunk_index += 1
            overlap_entries = max(
                1, overlap // 50
            )  # Approximate overlap in timestamp entries
            overlap_start = max(0, len(current_timestamps) - overlap_entries)

            current_timestamps = current_timestamps[overlap_start:]
            current_text = " ".join([ts.get("text", "") for ts in current_timestamps])
            start_pos = start_pos + len(current_text)

        # Add current entry
        current_text = test_text
        current_timestamps.append(timestamp_entry)

    # Yield final chunk
    if current_text:
        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=start_pos + len(current_text),
            token_count=estimate_token_count(current_text),
            has_timestamps=True,
            timestamp_range=(
                current_timestamps[0].get("start") if current_timestamps else None,
                current_timestamps[-1].get("end") if current_timestamps else None,
            ),
        )

        chunk = TranscriptChunk(
            text=current_text, metadata=metadata, timestamps=current_timestamps.copy()
        )

        if verify_context_limits(current_text, max_tokens):
            yield chunk
        else:
            logger.warning(
                f"Final chunk {chunk_index} exceeds context limits, yielding anyway"
            )
            yield chunk


def _extract_timestamps_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract timestamp information from raw transcript text.

    Supports formats like:
    - [00:01:23] Text here
    - 00:01:23 Text here
    - (1:23) Text here
    """
    timestamps = []

    # Pattern for various timestamp formats
    patterns = [
        r"\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.+?)(?=\[|\Z)",  # [HH:MM:SS]
        r"\[(\d{1,2}):(\d{2})\]\s*(.+?)(?=\[|\Z)",  # [MM:SS]
        r"(\d{1,2}):(\d{2}):(\d{2})\s+(.+?)(?=\d{1,2}:|\Z)",  # HH:MM:SS
        r"(\d{1,2}):(\d{2})\s+(.+?)(?=\d{1,2}:|\Z)",  # MM:SS
        r"\((\d{1,2}):(\d{2})\)\s*(.+?)(?=\(|\Z)",  # (MM:SS)
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            groups = match.groups()

            if len(groups) == 4:  # HH:MM:SS format
                hours, minutes, seconds, text_content = groups
                start_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            elif len(groups) == 3:  # MM:SS format
                minutes, seconds, text_content = groups
                start_time = int(minutes) * 60 + int(seconds)
            else:
                continue

            timestamps.append(
                {
                    "start": start_time,
                    "end": start_time + 30,  # Estimate 30 second duration
                    "text": text_content.strip(),
                }
            )

    return timestamps


def chunk_by_semantic_similarity(
    text: str, max_tokens: int = 8000
) -> Generator[TranscriptChunk, None, None]:
    """Chunk text using semantic similarity (placeholder implementation)."""
    logger.info("Semantic chunking not yet implemented, falling back to token chunking")
    for chunk in chunk_by_tokens(text, max_tokens):
        yield chunk


def smart_chunk_for_kg_generation(
    text: str, max_tokens: int = 4000
) -> Generator[TranscriptChunk, None, None]:
    """Smart chunking optimized for knowledge graph generation."""
    logger.info(
        "KG-optimized chunking not yet implemented, falling back to token chunking"
    )
    for chunk in chunk_by_tokens(text, max_tokens):
        yield chunk


# Backward compatibility functions that return lists
def chunk_by_tokens_list(
    text: str, max_tokens: int = 8000, overlap: int = 200
) -> List[str]:
    """Legacy function that returns a list instead of generator."""
    return [chunk.text for chunk in chunk_by_tokens(text, max_tokens, overlap)]


def chunk_by_semantic_similarity_list(text: str, max_tokens: int = 8000) -> List[str]:
    """Legacy function that returns a list instead of generator."""
    return [chunk.text for chunk in chunk_by_semantic_similarity(text, max_tokens)]


def smart_chunk_for_kg_generation_list(text: str, max_tokens: int = 4000) -> List[str]:
    """Legacy function that returns a list instead of generator."""
    return [chunk.text for chunk in smart_chunk_for_kg_generation(text, max_tokens)]
