"""Token counting utility using tiktoken."""

import logging
from typing import Dict

import tiktoken

logger = logging.getLogger(__name__)

# A mapping from model name to encoding.
# Add more models as needed.
ENCODING_FOR_MODEL = {
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a text string for a given model.

    Args:
        text: The text to count tokens for.
        model: The model name to use for tokenization.
               Defaults to "gpt-4".

    Returns:
        The number of tokens in the text.

    Raises:
        KeyError: If the model is not supported.
    """
    try:
        encoding_name = ENCODING_FOR_MODEL[model]
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        # If the model is not in our map, try to get the encoding for the model directly.
        # This will raise a KeyError if the model is not found by tiktoken.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as e:
            raise KeyError(
                f"Model '{model}' not found. Please use a supported model."
            ) from e

    return len(encoding.encode(text))


def count_gemini_tokens(text: str, model: str = "gemini-pro") -> int:
    """Count tokens for Gemini models."""
    # TODO: Implement Gemini-specific token counting
    logger.info(f"Counting Gemini tokens for model: {model}")
    # Rough estimation: similar to OpenAI
    return len(text) // 4


def count_tokens_all_models(text: str) -> Dict[str, int]:
    """Count tokens for all supported models."""
    return {
        "gemini_pro": count_gemini_tokens(text, "gemini-pro"),
    }
