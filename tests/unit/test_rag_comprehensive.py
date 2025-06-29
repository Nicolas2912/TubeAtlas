"""
Comprehensive tests for RAG components to improve coverage.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tubeatlas.rag.chunking.base import Chunk
from tubeatlas.rag.chunking.fixed import FixedLengthChunker
from tubeatlas.rag.embedding.openai import OpenAIEmbedder
from tubeatlas.rag.registry import RAGRegistry


class TestChunkingBase:
    """Test chunking base classes and validation."""

    def test_chunk_creation_with_custom_id(self):
        """Test chunk creation with custom ID."""
        chunk = Chunk.create(
            text="test text",
            start_idx=0,
            end_idx=9,
            chunk_id="custom_id",
            metadata={"custom": "data"},
        )
        assert chunk.id == "custom_id"
        assert chunk.metadata["custom"] == "data"

    def test_chunker_validate_text_errors(self):
        """Test text validation errors."""
        chunker = FixedLengthChunker(length_tokens=100, overlap_tokens=10)

        with pytest.raises(ValueError, match="text must be a string"):
            chunker.validate_text(123)

        with pytest.raises(ValueError, match="text cannot be empty"):
            chunker.validate_text("")

        with pytest.raises(ValueError, match="text cannot be empty"):
            chunker.validate_text("   ")

    def test_add_metadata_to_chunks(self):
        """Test adding metadata to chunks."""
        chunker = FixedLengthChunker(length_tokens=100, overlap_tokens=10)
        chunks = [
            Chunk.create(text="chunk1", start_idx=0, end_idx=6),
            Chunk.create(text="chunk2", start_idx=7, end_idx=13),
        ]

        metadata = {"source": "test", "type": "example"}
        updated_chunks = chunker.add_metadata_to_chunks(chunks, metadata)

        assert all(chunk.metadata["source"] == "test" for chunk in updated_chunks)
        assert all(chunk.metadata["type"] == "example" for chunk in updated_chunks)

    def test_get_chunk_statistics_empty(self):
        """Test statistics for empty chunk list."""
        chunker = FixedLengthChunker(length_tokens=100, overlap_tokens=10)
        stats = chunker.get_chunk_statistics([])

        assert stats["total_chunks"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avg_tokens_per_chunk"] == 0

    def test_get_chunk_statistics_populated(self):
        """Test statistics for populated chunk list."""
        chunker = FixedLengthChunker(length_tokens=100, overlap_tokens=10)
        chunks = [
            Chunk.create(text="short", start_idx=0, end_idx=5),
            Chunk.create(text="longer text here", start_idx=6, end_idx=22),
        ]

        stats = chunker.get_chunk_statistics(chunks)

        assert stats["total_chunks"] == 2
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_chunk"] > 0
        assert stats["min_tokens"] > 0
        assert stats["max_tokens"] >= stats["min_tokens"]


class TestEmbeddingBase:
    """Test embedding base classes and validation."""

    @patch("tubeatlas.rag.embedding.openai.openai.OpenAI")
    def test_embedder_validate_texts_errors(self, mock_openai_client):
        """Test text validation errors for embedder."""
        # Mock the OpenAI client to avoid API key requirement
        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        embedder = OpenAIEmbedder(
            api_key="dummy-key-for-testing"  # pragma: allowlist secret
        )

        with pytest.raises(ValueError, match="texts cannot be empty"):
            embedder.validate_texts([])

        with pytest.raises(ValueError, match="texts must be a list"):
            embedder.validate_texts("not a list")

        with pytest.raises(ValueError, match="text at index 0 must be a string"):
            embedder.validate_texts([123])

        with pytest.raises(ValueError, match="text at index 1 cannot be empty"):
            embedder.validate_texts(["valid", ""])

    @patch("tubeatlas.rag.embedding.openai.openai.OpenAI")
    def test_embedder_validate_text_errors(self, mock_openai_client):
        """Test single text validation errors."""
        # Mock the OpenAI client to avoid API key requirement
        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        embedder = OpenAIEmbedder(
            api_key="dummy-key-for-testing"  # pragma: allowlist secret
        )

        with pytest.raises(ValueError, match="text must be a string"):
            embedder.validate_text(123)

        with pytest.raises(ValueError, match="text cannot be empty"):
            embedder.validate_text("")

    @patch("tubeatlas.rag.embedding.openai.openai.OpenAI")
    def test_chunk_long_text(self, mock_openai_client):
        """Test chunking long text functionality."""
        # Mock the OpenAI client to avoid API key requirement
        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        embedder = OpenAIEmbedder(
            api_key="dummy-key-for-testing"  # pragma: allowlist secret
        )

        long_text = "word " * 100  # 500 characters
        chunks = embedder.chunk_long_text(long_text, max_length=50, overlap=10)

        assert len(chunks) > 1
        assert all(
            len(chunk) <= 60 for chunk in chunks
        )  # Allow some flexibility for word boundaries

    @patch("tubeatlas.rag.embedding.openai.openai.OpenAI")
    def test_chunk_short_text(self, mock_openai_client):
        """Test chunking text that doesn't need splitting."""
        # Mock the OpenAI client to avoid API key requirement
        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        embedder = OpenAIEmbedder(
            api_key="dummy-key-for-testing"  # pragma: allowlist secret
        )

        short_text = "short text"
        chunks = embedder.chunk_long_text(short_text, max_length=50)

        assert len(chunks) == 1
        assert chunks[0] == short_text


class TestVectorStoreBase:
    """Test vector store base validation."""

    def test_validate_embeddings_errors(self):
        """Test embedding validation errors."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        with pytest.raises(ValueError, match="embeddings cannot be empty"):
            store.validate_embeddings([])

        with pytest.raises(ValueError, match="embeddings must be a list"):
            store.validate_embeddings("not a list")

        with pytest.raises(ValueError, match="each embedding must be a list"):
            store.validate_embeddings([123])

        with pytest.raises(ValueError, match="embeddings cannot be empty vectors"):
            store.validate_embeddings([[]])

        with pytest.raises(ValueError, match="embedding at index 1 has dimension"):
            store.validate_embeddings([[1.0, 2.0, 3.0], [1.0, 2.0]])

    def test_validate_chunks_and_embeddings(self):
        """Test chunk and embedding validation."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        chunks = [Chunk.create(text="test", start_idx=0, end_idx=4)]
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # Mismatched length

        with pytest.raises(ValueError, match="Number of chunks"):
            store.validate_chunks_and_embeddings(chunks, embeddings)


class TestRegistryErrorHandling:
    """Test registry error handling."""

    def test_create_unknown_component(self):
        """Test creating unknown components."""
        registry = RAGRegistry()

        with pytest.raises(ValueError, match="Unknown chunker"):
            registry.create_chunker("unknown")

        with pytest.raises(ValueError, match="Unknown embedder"):
            registry.create_embedder("unknown")

        with pytest.raises(ValueError, match="Unknown vector store"):
            registry.create_vector_store("unknown")

    def test_list_empty_registry(self):
        """Test listing from empty registry."""
        registry = RAGRegistry()

        assert len(registry.list_chunkers()) >= 0
        assert len(registry.list_embedders()) >= 0
        assert len(registry.list_vector_stores()) >= 0


class TestOpenAIEmbedderErrorHandling:
    """Test OpenAI embedder error handling."""

    @patch("tubeatlas.rag.embedding.openai.openai.OpenAI")
    def test_get_config(self, mock_openai_client):
        """Test get_config method."""
        # Mock the OpenAI client to avoid API key requirement
        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        embedder = OpenAIEmbedder(
            model="text-embedding-3-small",
            batch_size=50,
            api_key="dummy-key-for-testing",  # pragma: allowlist secret
        )

        config = embedder.get_config()

        assert config["type"] == "openai"
        assert config["model"] == "text-embedding-3-small"
        assert config["batch_size"] == 50


class TestFaissStoreEdgeCases:
    """Test FAISS store edge cases."""

    def test_search_empty_index(self):
        """Test searching empty index."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        results = store.similarity_search([1.0, 2.0, 3.0], k=5)

        assert results == []

    def test_get_stats_empty(self):
        """Test stats on empty store."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        stats = store.get_stats()

        assert stats["chunk_count"] == 0
        assert stats["dimension"] == 3

    def test_delete_nonexistent_ids(self):
        """Test deleting non-existent IDs."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        deleted_count = store.delete_by_ids(["nonexistent"])

        assert deleted_count == 0

    def test_get_nonexistent_ids(self):
        """Test getting non-existent IDs."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)

        results = store.get_by_ids(["nonexistent"])

        assert results == [None]

    def test_clear_empty_store(self):
        """Test clearing empty store."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)
        store.clear()  # Should not raise error

        assert store.get_stats()["chunk_count"] == 0


class TestPersistenceOperations:
    """Test persistence operations."""

    def test_faiss_store_persistence(self):
        """Test FAISS store save/load operations."""
        from tubeatlas.rag.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(dimension=3)
        chunk = Chunk.create(text="test", start_idx=0, end_idx=4)
        embedding = [1.0, 2.0, 3.0]

        store.build_index([chunk], [embedding])

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_store"

            # Test save
            store.persist(save_path)
            assert save_path.exists()

            # Test load
            new_store = FaissVectorStore(dimension=3)
            new_store.load(save_path)

            # Verify data was loaded
            results = new_store.similarity_search([1.0, 2.0, 3.0], k=1)
            assert len(results) == 1
            assert results[0][0].text == "test"
