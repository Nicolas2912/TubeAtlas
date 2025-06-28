import sys
import types
from typing import List

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# FAISS stub (minimal implementation sufficient for FaissVectorStore tests)
# -----------------------------------------------------------------------------


def _create_faiss_stub():  # noqa: C901
    """Return a fake faiss module exposing minimal API used by FaissVectorStore."""
    faiss_stub = types.ModuleType("faiss")

    METRIC_INNER_PRODUCT = 0
    METRIC_L2 = 1

    class _BaseIndex:
        def __init__(self, dim: int):
            self.d = dim
            self.vectors: List[np.ndarray] = []
            self.ntotal = 0
            self.is_trained = True

        # FAISS expects float32 arrays
        def add(self, arr: np.ndarray):
            for row in arr:
                self.vectors.append(np.asarray(row, dtype=np.float32))
            self.ntotal = len(self.vectors)

        def _similarity(self, v1: np.ndarray, v2: np.ndarray, metric: str):
            if metric == "ip":
                return float(np.dot(v1, v2))
            else:  # l2 distance (negative for similarity purposes)
                return -float(np.linalg.norm(v1 - v2) ** 2)

        def search(self, query_arr: np.ndarray, k: int):
            metric = "ip"  # default, subclasses override for l2
            sims: List[np.ndarray] = []
            indices: List[np.ndarray] = []
            for q in query_arr:
                if not self.vectors:
                    sims.append([])
                    indices.append([])
                    continue
                scores = np.array(
                    [self._similarity(q, v, metric) for v in self.vectors]
                )
                top_idxs = np.argsort(-scores)[:k]
                sims.append(scores[top_idxs])
                indices.append(top_idxs)
            return np.array(sims), np.array(indices)

    class IndexFlatIP(_BaseIndex):
        pass

    class IndexFlatL2(_BaseIndex):
        def _similarity(self, v1, v2, metric="l2"):
            # return squared L2 distance as FAISS does; smaller is better
            return float(np.linalg.norm(v1 - v2) ** 2)

    # Simple wrappers for types not used in tests
    class IndexIVFFlat(IndexFlatIP):
        pass

    class IndexHNSWFlat(IndexFlatIP):
        def __init__(self, dim, M):
            super().__init__(dim)
            self.M = M

    # Serialization helpers (no-op)
    def write_index(index, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(index, f)

    def read_index(path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    # Attach attributes
    faiss_stub.IndexFlatIP = IndexFlatIP  # type: ignore[attr-defined]
    faiss_stub.IndexFlatL2 = IndexFlatL2  # type: ignore[attr-defined]
    faiss_stub.IndexIVFFlat = IndexIVFFlat  # type: ignore[attr-defined]
    faiss_stub.IndexHNSWFlat = IndexHNSWFlat  # type: ignore[attr-defined]
    faiss_stub.METRIC_INNER_PRODUCT = METRIC_INNER_PRODUCT  # type: ignore[attr-defined]
    faiss_stub.METRIC_L2 = METRIC_L2  # type: ignore[attr-defined]
    faiss_stub.write_index = write_index  # type: ignore[attr-defined]
    faiss_stub.read_index = read_index  # type: ignore[attr-defined]

    return faiss_stub


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_faiss(monkeypatch):
    """Automatically patch the `faiss` module for all tests in this file."""
    # Install stub before module import
    faiss_stub = _create_faiss_stub()
    sys.modules["faiss"] = faiss_stub
    yield
    # Cleanup
    sys.modules.pop("faiss", None)


# -----------------------------------------------------------------------------
# Tests for FaissVectorStore
# -----------------------------------------------------------------------------

from tubeatlas.rag.chunking.base import Chunk  # noqa: E402


def _make_chunk(idx: int, dim: int = 3, category: str = "default"):
    text = f"chunk {idx}"
    chunk = Chunk.create(text=text, start_idx=0, end_idx=len(text))
    chunk.metadata = {"category": category}
    embedding = [float(idx + 1)] + [0.0] * (dim - 1)
    return chunk, embedding


def test_build_and_search():
    from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

    # Prepare data
    chunks, embeddings = zip(*[_make_chunk(i) for i in range(3)])

    store = FaissVectorStore(dimension=3, index_type="flat", metric="cosine")
    store.build_index(list(chunks), list(embeddings))

    query = embeddings[1]  # Should retrieve chunk 2 with highest similarity
    results = store.similarity_search(query, k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.id == chunks[1].id
    # Similarity should be ~1 for identical normalized vectors
    assert pytest.approx(1.0, rel=1e-2) == score


def test_add_and_get_by_ids():
    from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

    chunk1, emb1 = _make_chunk(0)
    store = FaissVectorStore(dimension=3)
    store.build_index([chunk1], [emb1])

    # Add another chunk
    chunk2, emb2 = _make_chunk(1)
    store.add([chunk2], [emb2])

    retrieved = store.get_by_ids([chunk1.id, chunk2.id])
    assert retrieved == [chunk1, chunk2]


def test_filters_and_delete():
    from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

    chunk_a, emb_a = _make_chunk(0, category="news")
    chunk_b, emb_b = _make_chunk(1, category="sports")
    store = FaissVectorStore(dimension=3)
    store.build_index([chunk_a, chunk_b], [emb_a, emb_b])

    # Search with filter that matches only sports
    results = store.similarity_search(emb_b, k=5, filters={"category": "sports"})
    assert len(results) == 1 and results[0][0].id == chunk_b.id

    # Filter that matches none
    empty = store.similarity_search(emb_b, k=5, filters={"category": "tech"})
    assert empty == []

    # Delete and verify
    deleted = store.delete_by_ids([chunk_b.id])
    assert deleted == 1
    assert store.get_by_ids([chunk_b.id])[0] is None
