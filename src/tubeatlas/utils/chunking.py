"""Text chunking strategies."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_by_tokens(text: str, max_tokens: int = 8000, overlap: int = 200) -> List[str]:
    """Chunk text by token count with overlap."""
    # TODO: Implement proper token-based chunking
    logger.info(f"Chunking text into {max_tokens} token chunks with {overlap} overlap")

    # Simple character-based approximation for now
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap * chars_per_token

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a good breaking point (sentence end)
        break_point = text.rfind(".", start, end)
        if break_point == -1:
            break_point = end

        chunks.append(text[start : break_point + 1])
        start = break_point + 1 - overlap_chars

    return chunks


def chunk_by_semantic_similarity(text: str, max_tokens: int = 8000) -> List[str]:
    """Chunk text using semantic similarity."""
    # TODO: Implement semantic chunking using embeddings
    logger.info("Semantic chunking not yet implemented, falling back to token chunking")
    return chunk_by_tokens(text, max_tokens)


def smart_chunk_for_kg_generation(text: str, max_tokens: int = 4000) -> List[str]:
    """Smart chunking optimized for knowledge graph generation."""
    # TODO: Implement KG-optimized chunking (topic boundaries, entity preservation)
    logger.info(
        "KG-optimized chunking not yet implemented, falling back to token chunking"
    )
    return chunk_by_tokens(text, max_tokens)
