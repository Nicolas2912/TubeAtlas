# flake8: noqa
# mypy: ignore-errors

"""
OpenAI embeddings implementation with batching and rate limiting.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...utils.token_counter import TokenCounter
from .base import EmbedderInterface

try:
    import openai
    from tenacity import retry, stop_after_attempt, wait_exponential

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None

    # Create dummy decorator for when tenacity is not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbedderInterface):
    """
    OpenAI embeddings implementation with batching and rate limiting.

    Supports multiple OpenAI embedding models with automatic batching,
    rate limiting, and text chunking for long inputs.
    """

    # Model specifications
    MODEL_SPECS = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.00002,
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.00013,
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8192,
            "cost_per_1k_tokens": 0.0001,
        },
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        request_timeout: float = 30.0,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to environment variable)
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retry attempts
            request_timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai and tenacity are required for OpenAIEmbedder. "
                "Install with: pip install openai tenacity"
            )

        if model not in self.MODEL_SPECS:
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {list(self.MODEL_SPECS.keys())}"
            )

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"), timeout=request_timeout
        )

        # Model specifications
        self.model_spec = self.MODEL_SPECS[model]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self.validate_texts(texts)

        if not texts:
            return []

        # Handle long texts by chunking and averaging, but do it on the whole list
        final_embeddings = [[]] * len(texts)

        # Separate long and short texts, keeping original indices
        short_texts_with_indices = []
        long_texts_with_indices = []
        for i, text in enumerate(texts):
            if self._is_text_too_long(text):
                long_texts_with_indices.append((i, text))
            else:
                short_texts_with_indices.append((i, text))

        # Process short texts in batches
        if short_texts_with_indices:
            short_indices, short_texts = zip(*short_texts_with_indices)

            for i in range(0, len(short_texts), self.batch_size):
                batch_indices = short_indices[i : i + self.batch_size]
                batch_texts = short_texts[i : i + self.batch_size]

                batch_embeddings = self._embed_batch(list(batch_texts))

                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    final_embeddings[original_index] = embedding

                if i + self.batch_size < len(short_texts):
                    time.sleep(0.1)  # Small delay between batches

        # Process long texts one by one (as they require special handling)
        for original_index, long_text in long_texts_with_indices:
            final_embeddings[original_index] = self._embed_long_text(long_text)

        return final_embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        self.validate_text(text)

        if self._is_text_too_long(text):
            return self._embed_long_text(text)

        # Use the batch embedder for a single text for consistency
        return self._embed_batch([text])[0]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            # The API returns embeddings in the same order as the input
            sorted_embeddings = sorted(response.data, key=lambda e: e.index)
            return [item.embedding for item in sorted_embeddings]
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise

    def _embed_long_text(self, text: str) -> List[float]:
        """
        Embed long text by chunking and averaging embeddings.

        Args:
            text: Long text to embed

        Returns:
            Averaged embedding vector
        """
        # Chunk the text to fit within token limits
        chunks = self.chunk_long_text(text, self.model_spec["max_tokens"])

        if not chunks:
            # Fallback: truncate text
            truncated = TokenCounter.truncate_to_token_limit(
                text, self.model_spec["max_tokens"], self.model
            )
            return self._embed_batch([truncated])[0]

        # Get embeddings for all chunks
        chunk_embeddings = self._embed_batch(chunks)

        # Average the embeddings
        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]

        # Element-wise average using numpy
        return np.mean(chunk_embeddings, axis=0).tolist()

    def _is_text_too_long(self, text: str) -> bool:
        """
        Check if text exceeds the model's token limit.

        Args:
            text: Text to check

        Returns:
            True if text is too long
        """
        token_count = TokenCounter.count(text, self.model)
        return token_count > self.model_spec["max_tokens"]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.model_spec["dimensions"]

    def get_max_input_length(self) -> int:
        """Get the maximum input length in tokens for this model."""
        return self.model_spec["max_tokens"]

    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        return {
            "type": "openai",
            "model": self.model,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
            "dimensions": self.model_spec["dimensions"],
            "max_tokens": self.model_spec["max_tokens"],
        }

    def estimate_cost(self, texts: List[str]) -> float:
        """
        Estimate the cost of embedding the given texts.

        Args:
            texts: List of texts to estimate cost for

        Returns:
            Estimated cost in USD
        """
        total_tokens = sum(TokenCounter.count(text, self.model) for text in texts)
        cost_per_token = self.model_spec["cost_per_1k_tokens"] / 1000
        return total_tokens * cost_per_token
