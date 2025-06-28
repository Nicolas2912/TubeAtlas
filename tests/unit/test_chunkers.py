from tubeatlas.rag.chunking.fixed import FixedLengthChunker
from tubeatlas.rag.chunking.semantic import SemanticChunker
from tubeatlas.rag.embedding.base import EmbedderInterface
from tubeatlas.utils.token_counter import TokenCounter

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class DummyEmbedder(EmbedderInterface):
    """A simple embedder that returns orthogonal one-hot vectors."""

    def embed_texts(self, texts, model=None, batch_size=100, **kwargs):
        dim = 4  # tiny dimension is enough for test
        embeddings = []
        for idx, _ in enumerate(texts):
            vec = [0.0] * dim
            vec[idx % dim] = 1.0  # deterministic & non-zero
            embeddings.append(vec)
        return embeddings

    def embed_text(self, text, model=None, **kwargs):
        return self.embed_texts([text])[0]

    def get_embedding_dimension(self, model=None):
        return 4

    def get_max_input_length(self, model=None):
        return 2048

    def get_config(self):
        return {"type": "dummy", "dim": 4}


# ------------------------------------------------------------------
# Test data
# ------------------------------------------------------------------

TEXT = (
    "Sentence zero for testing purposes. "
    "Sentence one to follow. "
    "Sentence two comes next. "
    "Sentence three is also here. "
    "Finally sentence four closes the paragraph."
)

# ------------------------------------------------------------------
# FixedLengthChunker tests
# ------------------------------------------------------------------


def test_fixed_length_chunker_basic():
    chunker = FixedLengthChunker(
        length_tokens=30,
        overlap_tokens=0,
        model="gpt-3.5-turbo",
    )

    chunks = chunker.chunk(TEXT)

    # At least one chunk produced
    assert chunks, "No chunks returned"

    for chunk in chunks:
        # Token count should not exceed limit
        tok_count = TokenCounter.count(chunk.text, "gpt-3.5-turbo")
        assert tok_count <= 30
        # Indices should be consistent with original text
        assert TEXT[chunk.start_idx : chunk.end_idx].strip() == chunk.text
        # Metadata sanity check
        assert chunk.metadata["chunker"] == "fixed"


# ------------------------------------------------------------------
# SemanticChunker tests
# ------------------------------------------------------------------


def test_semantic_chunker_with_dummy_embedder():
    embedder = DummyEmbedder()
    chunker = SemanticChunker(
        embedder=embedder,
        similarity_threshold=0.3,  # encourage splitting
        max_chunk_tokens=100,
        min_chunk_tokens=5,
        token_model="gpt-3.5-turbo",
    )

    chunks = chunker.chunk(TEXT)

    # Should create multiple chunks (given low similarity threshold)
    assert len(chunks) >= 2

    for chunk in chunks:
        # Metadata contains semantic info and embedder config
        meta = chunk.metadata
        assert meta["chunker"] == "semantic"
        assert "embedder_config" in meta
        # Token count obeys limits
        assert TokenCounter.count(chunk.text, "gpt-3.5-turbo") <= 100
