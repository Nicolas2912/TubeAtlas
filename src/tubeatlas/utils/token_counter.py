"""Token counting utilities."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def count_openai_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens for OpenAI models."""
    # TODO: Implement tiktoken-based token counting
    logger.info(f"Counting OpenAI tokens for model: {model}")
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4


def count_gemini_tokens(text: str, model: str = "gemini-pro") -> int:
    """Count tokens for Gemini models."""
    # TODO: Implement Gemini-specific token counting
    logger.info(f"Counting Gemini tokens for model: {model}")
    # Rough estimation: similar to OpenAI
    return len(text) // 4


def count_tokens_all_models(text: str) -> Dict[str, int]:
    """Count tokens for all supported models."""
    return {
        "openai_gpt35": count_openai_tokens(text, "gpt-3.5-turbo"),
        "openai_gpt4": count_openai_tokens(text, "gpt-4"),
        "gemini_pro": count_gemini_tokens(text, "gemini-pro"),
    }
