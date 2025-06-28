"""
Token counting utility for various language models using tiktoken.
"""

import functools
import logging
from typing import Dict, List, Optional

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    A utility class for counting tokens for various models using tiktoken.

    This class provides a centralized way to handle token counting,
    supporting different encodings for different models and caching
    tokenizer instances for performance.
    """

    # Mapping from model prefix to encoding
    MODEL_TO_ENCODING = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "code-davinci-002": "p50k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }

    # Cache for tokenizer instances
    _tokenizers: Dict[str, "tiktoken.Encoding"] = {}

    @classmethod
    def get_encoding_for_model(cls, model: str) -> Optional["tiktoken.Encoding"]:
        """Get the encoding for a given model, falling back to a default."""
        if not TIKTOKEN_AVAILABLE:
            return None

        # Find the best matching encoding
        encoding_name = "cl100k_base"  # Default
        for prefix, name in cls.MODEL_TO_ENCODING.items():
            if model.startswith(prefix):
                encoding_name = name
                break
        else:
            logger.warning(
                f"Unknown model '{model}', falling back to {encoding_name} encoding"
            )

        # Use cached tokenizer if available
        if encoding_name in cls._tokenizers:
            return cls._tokenizers[encoding_name]

        try:
            encoding = tiktoken.get_encoding(encoding_name)
            cls._tokenizers[encoding_name] = encoding
            return encoding
        except Exception as e:
            logger.error(f"Failed to get encoding '{encoding_name}': {e}")
            return None

    # Provide legacy _get_encoding with caching for backward compatibility
    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _get_encoding(model: str):
        """Legacy cached encoding getter for backward compatibility with tests."""
        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install with: pip install tiktoken"
            )
        encoding = TokenCounter.get_encoding_for_model(model)
        if encoding is None:
            raise ImportError("Failed to obtain encoding; tiktoken may be missing")
        return encoding

    # ---------------------------------------------------------------------
    # Public utility methods
    # ---------------------------------------------------------------------

    @classmethod
    def count(cls, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text for a specific model, with chunking for long input."""
        if not text:
            return 0

        # Fallback if tiktoken not available
        if not TIKTOKEN_AVAILABLE:
            return max(1, len(text) // 4)

        # Process in chunks to avoid memory issues
        if len(text) > 100_000:
            return cls._count_chunked(text, model)

        encoding = cls._get_encoding(model)
        return len(encoding.encode(text))

    @classmethod
    def _count_chunked(cls, text: str, model: str, chunk_size: int = 50_000) -> int:
        """Count tokens in very long text by processing in chunks."""
        if not TIKTOKEN_AVAILABLE:
            return max(1, len(text) // 4)

        encoding = cls._get_encoding(model)
        total_tokens = 0
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            total_tokens += len(encoding.encode(chunk))
        return total_tokens

    @classmethod
    def encode(cls, text: str, model: str = "gpt-3.5-turbo") -> List[int]:
        """Encode text to tokens for a specific model."""
        if not text:
            return []
        if not TIKTOKEN_AVAILABLE:
            return []
        encoding = cls._get_encoding(model)
        return encoding.encode(text)

    @classmethod
    def decode(cls, tokens: List[int], model: str = "gpt-3.5-turbo") -> str:
        """Decode tokens back to text for a specific model."""
        if not tokens:
            return ""
        if not TIKTOKEN_AVAILABLE:
            return ""
        encoding = cls._get_encoding(model)
        return encoding.decode(tokens)

    @classmethod
    def truncate_to_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: str = "gpt-3.5-turbo",
    ) -> str:
        """Truncate text to a maximum number of tokens for the given model."""
        if not text:
            return ""
        if not TIKTOKEN_AVAILABLE:
            return text[: max_tokens * 4]
        encoding = cls._get_encoding(model)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Standalone function for backward compatibility.
    Delegates to the TokenCounter class.
    """
    return TokenCounter.count(text, model)


def main():  # noqa: C901
    """
    Simple CLI demo for TokenCounter utility.

    Examples:
        python -m tubeatlas.utils.token_counter "Hello world"
        python -m tubeatlas.utils.token_counter "Hello world" --model gpt-4
        echo "Hello world" | python -m tubeatlas.utils.token_counter
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Count tokens in text using OpenAI tiktoken",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world!"
  %(prog)s "Hello world!" --model gpt-4
  %(prog)s --file input.txt
  echo "Hello world!" | %(prog)s
        """,
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to count tokens for (if not provided, reads from stdin)",
    )

    parser.add_argument(
        "--model",
        "-m",
        default="gpt-3.5-turbo",
        help="Model to use for tokenization (default: gpt-3.5-turbo)",
    )

    parser.add_argument(
        "--file", "-f", help="Read text from file instead of command line"
    )

    parser.add_argument(
        "--encode", action="store_true", help="Show token IDs instead of just count"
    )

    parser.add_argument(
        "--truncate", "-t", type=int, help="Truncate text to specified token limit"
    )

    args = parser.parse_args()

    # Get text from various sources
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        # Read from stdin
        try:
            text = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit(1)

    if not text.strip():
        print("No text provided", file=sys.stderr)
        sys.exit(1)

    try:
        if args.truncate:
            # Truncate text
            truncated = TokenCounter.truncate_to_token_limit(
                text, args.truncate, args.model
            )
            print(f"Original tokens: {TokenCounter.count(text, args.model)}")
            print(f"Truncated tokens: {TokenCounter.count(truncated, args.model)}")
            print(f"Truncated text:\n{truncated}")
        elif args.encode:
            # Show token IDs
            tokens = TokenCounter.encode(text, args.model)
            print(f"Token count: {len(tokens)}")
            print(f"Token IDs: {tokens}")
        else:
            # Just count tokens
            count = TokenCounter.count(text, args.model)
            print(f"Token count: {count}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
