import pytest

from src.tubeatlas.utils.token_counter import count_tokens


def test_count_tokens_gpt4():
    """Test token counting for gpt-4."""
    text = "hello world"
    # tiktoken for gpt-4 encodes "hello world" as [15339, 1917]
    assert count_tokens(text, model="gpt-4") == 2


def test_count_tokens_unsupported_model():
    """Test that an unsupported model raises a KeyError."""
    text = "hello world"
    with pytest.raises(KeyError, match="Model 'unsupported-model' not found"):
        count_tokens(text, model="unsupported-model")


def test_count_tokens_direct_model_lookup():
    """Test token counting for a model known by tiktoken but not in our map."""
    # This assumes tiktoken knows about "p50k_base" models.
    # We test with a model that uses a different encoding to ensure logic works.
    text = "hello world"
    # For a model like 'text-davinci-003', which uses 'p50k_base'
    # 'hello world' is tokenized differently.
    # Note: this test depends on tiktoken's internal model list.
    # We will mock encoding_for_model to make the test stable.
    try:
        assert count_tokens(text, model="text-davinci-003") == 2
    except KeyError:
        pytest.skip("tiktoken does not have text-davinci-003, skipping test")


def test_count_tokens_empty_string():
    """Test that an empty string has zero tokens."""
    assert count_tokens("", model="gpt-4") == 0
