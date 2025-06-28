from typing import Any, Dict, List, Optional, Tuple

import pytest

from tubeatlas.rag.chunking.base import Chunk, ChunkerInterface
from tubeatlas.rag.embedding.base import EmbedderInterface

# Import pipelines early to satisfy flake8 E402
from tubeatlas.rag.pipeline import IngestPipeline, RetrievalPipeline
from tubeatlas.rag.registry import get_registry
from tubeatlas.rag.vector_store.base import VectorStoreInterface

# -----------------------------------------------------------------------------
# Dummy component implementations
# -----------------------------------------------------------------------------


class DummyEmbedder(EmbedderInterface):
    """Very simple embedder that maps text to 3-dimensional vector."""

    def __init__(self, **kwargs):
        self.dim = 3

    def embed_text(self, text: str) -> List[float]:  # type: ignore[override]
        return [float(len(text)), 0.0, 0.0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return [self.embed_text(t) for t in texts]

    # Helpers
    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_max_input_length(self) -> int:
        return 1000

    def get_config(self):
        return {}


class DummyChunker(ChunkerInterface):
    """Returns the whole text as a single chunk."""

    def __init__(self, **kwargs):
        self.model = "dummy"

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        metadata = metadata or {}
        chunk = Chunk.create(
            text=text, start_idx=0, end_idx=len(text), model=self.model
        )
        chunk.metadata.update(metadata)
        return [chunk]

    def get_config(self):
        return {}


class DummyVectorStore(VectorStoreInterface):
    """Stores chunks in-memory and returns naïve similarity based on first element."""

    def __init__(self, **kwargs):
        self.data: List[Tuple[Chunk, List[float]]] = []
        self.dimension = 3

    def build_index(self, chunks, embeddings, **kwargs):  # type: ignore[override]
        self.clear()
        self.add(chunks, embeddings)

    def add(self, chunks, embeddings, **kwargs):  # type: ignore[override]
        for c, e in zip(chunks, embeddings):
            self.data.append((c, e))

    def similarity_search(self, query_embedding, k=5, filters=None, **kwargs):  # type: ignore[override]
        results = []
        for chunk, emb in self.data:
            if filters and not all(
                chunk.metadata.get(k) == v for k, v in filters.items()
            ):
                continue
            # very simple score: inverse absolute distance on first coord
            score = 1.0 / (1.0 + abs(query_embedding[0] - emb[0]))
            results.append((chunk, score))
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def get_by_ids(self, chunk_ids):  # type: ignore[override]
        mapping = {c.id: c for c, _ in self.data}
        return [mapping.get(cid) for cid in chunk_ids]

    def delete_by_ids(self, chunk_ids):  # type: ignore[override]
        before = len(self.data)
        self.data = [pair for pair in self.data if pair[0].id not in chunk_ids]
        return before - len(self.data)

    def persist(self, path):  # type: ignore[override]
        pass

    def load(self, path):  # type: ignore[override]
        pass

    def get_stats(self):  # type: ignore[override]
        return {"count": len(self.data)}

    def clear(self):  # type: ignore[override]
        self.data.clear()


# -----------------------------------------------------------------------------
# Fixtures to patch registry
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def register_dummy_components():
    registry = get_registry()
    # Ensure unique names to avoid collisions
    registry.register_chunker("dummy", DummyChunker)
    registry.register_embedder("dummy", DummyEmbedder)
    registry.register_vector_store("dummy", DummyVectorStore)
    yield
    # Cleanup registry after tests
    registry.clear_registry()


# -----------------------------------------------------------------------------
# Tests for Registry
# -----------------------------------------------------------------------------


def test_registry_creation_and_listing():
    registry = get_registry()

    chunker = registry.create_chunker("dummy")
    embedder = registry.create_embedder("dummy")
    store = registry.create_vector_store("dummy")

    assert isinstance(chunker, DummyChunker)
    assert isinstance(embedder, DummyEmbedder)
    assert isinstance(store, DummyVectorStore)

    # Listing APIs include the dummy entries
    assert "dummy" in registry.list_chunkers()
    assert "dummy" in registry.list_embedders()
    assert "dummy" in registry.list_vector_stores()


# -----------------------------------------------------------------------------
# Tests for IngestPipeline and RetrievalPipeline
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_and_retrieve_flow():
    # Prepare documents
    docs = [
        {"id": "doc1", "text": "hello world", "category": "greet"},
        {"id": "doc2", "text": "foo bar", "category": "other"},
    ]

    pipeline = IngestPipeline(
        chunker_name="dummy",
        embedder_name="dummy",
        vector_store_name="dummy",
        batch_size=1,
    )

    result = await pipeline.ingest_documents(docs, metadata_fields=["category"])

    assert result["success"]
    stats = result["stats"]
    assert stats["documents_processed"] == 2
    assert stats["chunks_created"] == 2  # one chunk per doc
    assert stats["embeddings_generated"] == 2

    # Use same store for retrieval
    retriever = RetrievalPipeline(
        embedder_name="dummy",
        vector_store=pipeline.vector_store,
        default_k=1,
    )

    query = "hello"
    retrieval_results = await retriever.retrieve(query, k=1)

    assert len(retrieval_results) == 1
    top_chunk, score = retrieval_results[0]
    # Based on dummy similarity, doc2 should be closest (length difference smallest)
    assert top_chunk.metadata["document_id"].startswith("doc2")
    assert 0.0 <= score <= 1.0
