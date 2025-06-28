"""
Tests for the TokenCounter utility.
"""

from unittest.mock import patch

import pytest

from tubeatlas.utils.token_counter import TokenCounter


class TestTokenCounter:
    """Test cases for TokenCounter class."""

    def test_count_empty_text(self):
        """Test counting tokens for empty text."""
        assert TokenCounter.count("") == 0
        assert TokenCounter.count("   ") == 1  # Whitespace still has tokens

    def test_count_simple_text(self):
        """Test counting tokens for simple text."""
        text = "Hello world"
        count = TokenCounter.count(text)
        assert isinstance(count, int)
        assert count > 0

    def test_count_with_different_models(self):
        """Test counting with different model specifications."""
        text = "Hello world"

        # Test with default model
        count_default = TokenCounter.count(text)

        # Test with explicit gpt-3.5-turbo
        count_gpt35 = TokenCounter.count(text, "gpt-3.5-turbo")

        # Test with gpt-4
        count_gpt4 = TokenCounter.count(text, "gpt-4")

        assert count_default == count_gpt35
        assert isinstance(count_gpt4, int)
        assert count_gpt4 > 0

    def test_count_long_text_chunking(self):
        """Test that very long texts are processed in chunks."""
        # Create a text longer than 100K characters
        long_text = "Hello world! " * 10000  # ~130K characters

        count = TokenCounter.count(long_text)
        assert isinstance(count, int)
        assert count > 0

    def test_encode_simple_text(self):
        """Test encoding text to tokens."""
        text = "Hello"
        tokens = TokenCounter.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)

    def test_encode_empty_text(self):
        """Test encoding empty text."""
        assert TokenCounter.encode("") == []

    def test_decode_tokens(self):
        """Test decoding tokens back to text."""
        original_text = "Hello world"
        tokens = TokenCounter.encode(original_text)
        decoded_text = TokenCounter.decode(tokens)

        assert decoded_text == original_text

    def test_decode_empty_tokens(self):
        """Test decoding empty token list."""
        assert TokenCounter.decode([]) == ""

    def test_encode_decode_roundtrip(self):
        """Test that encode/decode is a perfect roundtrip."""
        texts = [
            "Hello world",
            "This is a longer text with punctuation!",
            "Unicode: 你好世界",
            "Numbers: 12345",
            "Special chars: @#$%^&*()",
        ]

        for text in texts:
            tokens = TokenCounter.encode(text)
            decoded = TokenCounter.decode(tokens)
            assert decoded == text

    def test_truncate_to_token_limit(self):
        """Test truncating text to token limit."""
        text = "This is a longer text that will be truncated"
        max_tokens = 5

        truncated = TokenCounter.truncate_to_token_limit(text, max_tokens)
        truncated_count = TokenCounter.count(truncated)

        assert truncated_count <= max_tokens
        assert len(truncated) < len(text)

    def test_truncate_short_text(self):
        """Test truncating text that's already under limit."""
        text = "Short"
        max_tokens = 100

        truncated = TokenCounter.truncate_to_token_limit(text, max_tokens)
        assert truncated == text

    def test_truncate_empty_text(self):
        """Test truncating empty text."""
        assert TokenCounter.truncate_to_token_limit("", 10) == ""

    def test_caching_behavior(self):
        """Test that encoding is cached for performance."""
        # Clear any existing cache
        TokenCounter._get_encoding.cache_clear()

        # First call should populate cache
        TokenCounter.count("test", "gpt-3.5-turbo")
        cache_info_1 = TokenCounter._get_encoding.cache_info()

        # Second call should hit cache
        TokenCounter.count("another test", "gpt-3.5-turbo")
        cache_info_2 = TokenCounter._get_encoding.cache_info()

        # Cache hits should increase
        assert cache_info_2.hits > cache_info_1.hits

    def test_unknown_model_fallback(self):
        """Test that unknown models fall back to cl100k_base."""
        text = "Hello world"

        # This should not raise an error even with unknown model
        count = TokenCounter.count(text, "unknown-model-xyz")
        assert isinstance(count, int)
        assert count > 0

    def test_missing_tiktoken_dependency(self):
        """Test behavior when tiktoken is not installed."""
        # Test by patching both tiktoken and TIKTOKEN_AVAILABLE
        with (
            patch("tubeatlas.utils.token_counter.tiktoken", None),
            patch("tubeatlas.utils.token_counter.TIKTOKEN_AVAILABLE", False),
        ):
            # Clear the cache first
            TokenCounter._get_encoding.cache_clear()
            with pytest.raises(ImportError, match="tiktoken is required"):
                TokenCounter._get_encoding("gpt-3.5-turbo")

    def test_chunked_counting_approximate_consistency(self):
        """Test that chunked counting gives approximately same result as regular counting."""
        # Note: Due to how tiktoken handles token boundaries, chunked counting
        # may differ slightly from normal counting, but should be close
        text = "This is a test sentence. " * 1000  # ~25K characters

        # Count normally
        normal_count = TokenCounter.count(text)

        # Count with forced chunking
        chunked_count = TokenCounter._count_chunked(
            text, "gpt-3.5-turbo", chunk_size=5000
        )

        # Results should be approximately the same (within 1% tolerance)
        tolerance = max(1, normal_count * 0.01)  # 1% tolerance, minimum 1 token
        assert abs(normal_count - chunked_count) <= tolerance

    def test_different_chunk_sizes_approximate(self):
        """Test chunked counting with different chunk sizes gives similar results."""
        text = "This is a test. " * 1000

        count_1 = TokenCounter._count_chunked(text, "gpt-3.5-turbo", chunk_size=1000)
        count_2 = TokenCounter._count_chunked(text, "gpt-3.5-turbo", chunk_size=5000)

        # Different chunk sizes should give approximately the same total count
        # Allow for small differences due to token boundary effects
        tolerance = max(1, count_1 * 0.01)  # 1% tolerance
        assert abs(count_1 - count_2) <= tolerance


class TestTokenCounterCLI:
    """Test cases for TokenCounter CLI functionality."""

    @patch("sys.argv", ["token_counter", "Hello world"])
    @patch("builtins.print")
    def test_cli_basic_usage(self, mock_print):
        """Test basic CLI usage."""
        from tubeatlas.utils.token_counter import main

        try:
            main()
        except SystemExit:
            pass  # CLI calls sys.exit, which is expected

        # Check that output was printed
        assert mock_print.called

    @patch("sys.argv", ["token_counter", "Hello world", "--model", "gpt-4"])
    @patch("builtins.print")
    def test_cli_with_model(self, mock_print):
        """Test CLI with specific model."""
        from tubeatlas.utils.token_counter import main

        try:
            main()
        except SystemExit:
            pass

        assert mock_print.called

    @patch("sys.argv", ["token_counter", "Hello world", "--encode"])
    @patch("builtins.print")
    def test_cli_with_encode(self, mock_print):
        """Test CLI with encode flag."""
        from tubeatlas.utils.token_counter import main

        try:
            main()
        except SystemExit:
            pass

        assert mock_print.called

    @patch("sys.argv", ["token_counter", "Hello world", "--truncate", "3"])
    @patch("builtins.print")
    def test_cli_with_truncate(self, mock_print):
        """Test CLI with truncate option."""
        from tubeatlas.utils.token_counter import main

        try:
            main()
        except SystemExit:
            pass

        assert mock_print.called


class TestTokenCounterIntegration:
    """Integration tests for TokenCounter with real tiktoken."""

    def test_real_token_counts(self):
        """Test with real tiktoken to verify reasonable token counts."""
        # These tests verify that our token counting gives reasonable results
        test_cases = [
            ("hello", 1),  # Simple word should be 1 token
            ("hello world", 2),  # Two words should be ~2 tokens
            ("The quick brown fox", 4),  # Should be ~4 tokens
        ]

        for text, expected_approx in test_cases:
            count = TokenCounter.count(text)
            # Allow some tolerance since exact counts depend on tokenizer
            assert count >= expected_approx - 1
            assert count <= expected_approx + 2

    def test_consistency_across_calls(self):
        """Test that multiple calls with same input give same result."""
        text = "This is a consistency test"

        count1 = TokenCounter.count(text)
        count2 = TokenCounter.count(text)
        count3 = TokenCounter.count(text, "gpt-3.5-turbo")

        assert count1 == count2 == count3

    def test_model_differences(self):
        """Test that different models may give different token counts."""
        text = "This is a test of model differences"

        count_gpt35 = TokenCounter.count(text, "gpt-3.5-turbo")
        count_gpt4 = TokenCounter.count(text, "gpt-4")

        # Both should be positive integers
        assert isinstance(count_gpt35, int) and count_gpt35 > 0
        assert isinstance(count_gpt4, int) and count_gpt4 > 0

        # They might be the same or different, but both should be reasonable
        assert abs(count_gpt35 - count_gpt4) <= max(count_gpt35, count_gpt4) * 0.5
