from unittest.mock import MagicMock

from tubeatlas.rag.embedding.openai import OpenAIEmbedder

# ------------------------------------------------------------------
# Helper – create mock OpenAI client
# ------------------------------------------------------------------


def mock_openai(monkeypatch, dim=1536):
    """Patch tubeatlas.rag.embedding.openai.openai.OpenAI to a dummy client."""

    # Build dummy embedding object factory
    def make_embedding(index: int):
        emb = MagicMock()
        emb.index = index
        emb.embedding = [float(index)] * dim  # deterministic vector
        return emb

    dummy_client = MagicMock()

    def create_embeddings(model, input, **kwargs):
        data = [make_embedding(i) for i, _ in enumerate(input)]
        resp = MagicMock()
        resp.data = data
        return resp

    dummy_client.embeddings.create.side_effect = create_embeddings

    # Patch the OpenAI constructor used in the module
    monkeypatch.setattr(
        "tubeatlas.rag.embedding.openai.openai.OpenAI", lambda **kwargs: dummy_client
    )

    return dummy_client


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_embed_texts_batches(monkeypatch):
    """OpenAIEmbedder should batch requests respecting batch_size."""
    dummy_client = mock_openai(monkeypatch)

    embedder = OpenAIEmbedder(batch_size=2)  # force multiple batches

    texts = [f"text {i}" for i in range(5)]  # 5 texts -> 3 batches (2,2,1)
    embeddings = embedder.embed_texts(texts)

    # Correct number of embeddings returned in same order
    assert len(embeddings) == len(texts)
    # Validate first and last embeddings are list[float] of expected dimension
    dim = embedder.get_embedding_dimension()
    assert isinstance(embeddings[0], list) and len(embeddings[0]) == dim
    assert isinstance(embeddings[4], list) and len(embeddings[4]) == dim

    # create called 3 times
    assert dummy_client.embeddings.create.call_count == 3


def test_embed_text_single(monkeypatch):
    """embed_text should wrap single call correctly and return vector."""
    mock_openai(monkeypatch)
    embedder = OpenAIEmbedder(batch_size=10)

    vec = embedder.embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) == embedder.get_embedding_dimension()


def test_metadata_properties(monkeypatch):
    """Check dimension and max input length helpers."""
    mock_openai(monkeypatch)
    embedder = OpenAIEmbedder()

    spec = embedder.MODEL_SPECS[embedder.model]
    assert embedder.get_embedding_dimension() == spec["dimensions"]
    assert embedder.get_max_input_length() == spec["max_tokens"]
