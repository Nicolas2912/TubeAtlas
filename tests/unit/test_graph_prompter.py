"""
Tests for GraphPrompter knowledge graph extraction.
"""

# type: ignore

from unittest.mock import Mock, patch

import pytest

from src.tubeatlas.rag.chunking.base import Chunk
from src.tubeatlas.rag.graph_extraction import (
    GraphPrompter,
    GraphPrompterConfig,
    Triple,
)


# Mock classes for LangChain components
class MockNode:
    """Mock node for testing."""

    def __init__(self, id: str, type: str = "ENTITY"):
        self.id = id
        self.type = type


class MockRelationship:
    """Mock relationship for testing."""

    def __init__(self, source_id: str, target_id: str, relation_type: str):
        self.source = MockNode(source_id)
        self.target = MockNode(target_id)
        self.type = relation_type


class MockGraphDocument:
    """Mock graph document for testing."""

    def __init__(self, relationships=None, nodes=None):
        self.relationships = relationships or []
        self.nodes = nodes or []


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return GraphPrompterConfig(
        primary_model="gpt-3.5-turbo",
        fallback_model="gpt-4",
        primary_max_tokens=100,  # Low for testing fallback
        fallback_max_tokens=500,
        batch_size=2,
        max_retries=2,
        retry_delay=0.1,  # Fast for testing
        additional_instructions="Test instructions",
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk.create(
            text="John teaches machine learning at Stanford University.",
            start_idx=0,
            end_idx=50,
            chunk_id="chunk_1",
            metadata={"speaker": "narrator", "timestamp": "00:01:00"},
        ),
        Chunk.create(
            text="Python is a programming language used for data science.",
            start_idx=51,
            end_idx=105,
            chunk_id="chunk_2",
            metadata={"speaker": "narrator", "timestamp": "00:02:00"},
        ),
    ]


@pytest.fixture
def large_chunk():
    """Create a large chunk that would trigger fallback."""
    # Create a chunk with high token count to trigger fallback
    large_text = "This is a very long text. " * 50  # Should exceed primary_max_tokens
    return Chunk.create(
        text=large_text,
        start_idx=0,
        end_idx=len(large_text),
        chunk_id="large_chunk",
        metadata={"speaker": "narrator", "timestamp": "00:03:00"},
    )


class TestTriple:
    """Test the Triple dataclass."""

    def test_triple_creation(self):
        """Test creating a Triple instance."""
        triple = Triple(
            subject="John",
            predicate="teaches",
            object="machine learning",
            confidence=0.9,
            provenance={"chunk_id": "test_chunk"},
        )

        assert triple.subject == "John"
        assert triple.predicate == "teaches"
        assert triple.object == "machine learning"
        assert triple.confidence == 0.9
        assert triple.provenance == {"chunk_id": "test_chunk"}

    def test_triple_to_dict(self):
        """Test converting Triple to dictionary."""
        triple = Triple(
            subject="Python",
            predicate="is",
            object="programming language",
            confidence=0.95,
            provenance={"source": "test"},
        )

        result = triple.to_dict()
        expected = {
            "subject": "Python",
            "predicate": "is",
            "object": "programming language",
            "confidence": 0.95,
            "provenance": {"source": "test"},
        }

        assert result == expected

    def test_triple_to_dict_minimal(self):
        """Test converting Triple to dictionary without optional fields."""
        triple = Triple(subject="Subject", predicate="predicate", object="Object")

        result = triple.to_dict()
        expected = {"subject": "Subject", "predicate": "predicate", "object": "Object"}

        assert result == expected


class TestGraphPrompterConfig:
    """Test the GraphPrompterConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GraphPrompterConfig()

        assert config.primary_model == "gpt-3.5-turbo"
        assert config.fallback_model == "gpt-4"
        assert config.primary_max_tokens == 4000
        assert config.fallback_max_tokens == 8000
        assert config.temperature == 0.0
        assert config.strict_mode is True
        assert config.batch_size == 5

    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_from_yaml(self, mock_yaml_load, mock_open):
        """Test loading configuration from YAML file."""
        mock_yaml_data = {
            "graph_prompter": {
                "primary_model": "custom-model",
                "temperature": 0.5,
                "batch_size": 10,
            }
        }
        mock_yaml_load.return_value = mock_yaml_data

        config = GraphPrompterConfig.from_yaml("test.yaml")

        assert config.primary_model == "custom-model"
        assert config.temperature == 0.5
        assert config.batch_size == 10
        # Other values should be defaults
        assert config.fallback_model == "gpt-4"


class TestGraphPrompter:
    """Test the GraphPrompter class."""

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    def test_initialization(self, mock_transformer, mock_openai, sample_config):
        """Test GraphPrompter initialization."""
        prompter = GraphPrompter(sample_config)

        # Check that OpenAI models were created
        assert mock_openai.call_count == 2  # Primary and fallback
        assert mock_transformer.call_count == 2  # Primary and fallback transformers

        assert prompter.config == sample_config

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    def test_initialization_without_config(self, mock_transformer, mock_openai):
        """Test GraphPrompter initialization with default config."""
        with patch.object(GraphPrompter, "_load_default_config") as mock_load:
            mock_load.return_value = GraphPrompterConfig()
            prompter = GraphPrompter()

            assert mock_load.called
            assert prompter.config is not None

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_single_chunk(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test extracting triples from a single chunk."""
        # Setup mock relationships
        mock_relationships = [
            MockRelationship("John", "machine learning", "TEACHES"),
            MockRelationship("John", "Stanford University", "WORKS_AT"),
        ]
        mock_graph_doc = MockGraphDocument(relationships=mock_relationships)

        # Mock the transformer
        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.return_value = [
            mock_graph_doc
        ]

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        # Test with single chunk (should use primary model)
        chunk = sample_chunks[0]  # This has low token count
        result = await prompter.aextract_triples([chunk])

        assert len(result) == 2

        # Check first triple
        assert result[0].subject == "John"
        assert result[0].predicate == "teaches"  # Should be lowercase and formatted
        assert result[0].object == "machine learning"
        assert result[0].provenance["chunk_id"] == "chunk_1"
        assert result[0].provenance["speaker"] == "narrator"

        # Check second triple
        assert result[1].subject == "John"
        assert result[1].predicate == "works at"  # Underscore should be replaced
        assert result[1].object == "Stanford University"

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_fallback_model(
        self, mock_transformer, mock_openai, sample_config, large_chunk
    ):
        """Test that large chunks trigger fallback model."""
        # Setup mock relationships
        mock_relationships = [MockRelationship("Topic", "content", "DISCUSSES")]
        mock_graph_doc = MockGraphDocument(relationships=mock_relationships)

        # Mock the fallback transformer
        mock_fallback_transformer = Mock()
        mock_fallback_transformer.convert_to_graph_documents.return_value = [
            mock_graph_doc
        ]

        prompter = GraphPrompter(sample_config)
        prompter._fallback_transformer = mock_fallback_transformer

        # Force the chunk to have high token count
        large_chunk.token_count = 200  # Exceeds primary_max_tokens (100)

        result = await prompter.aextract_triples([large_chunk])

        assert len(result) == 1
        assert result[0].subject == "Topic"
        assert result[0].predicate == "discusses"
        assert result[0].object == "content"

        # Verify fallback transformer was used
        mock_fallback_transformer.convert_to_graph_documents.assert_called_once()

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_batch_processing(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test batch processing of multiple chunks."""
        # Setup mock relationships for different chunks
        mock_relationships_1 = [MockRelationship("John", "machine learning", "TEACHES")]
        mock_relationships_2 = [
            MockRelationship("Python", "programming language", "IS")
        ]

        mock_graph_doc_1 = MockGraphDocument(relationships=mock_relationships_1)
        mock_graph_doc_2 = MockGraphDocument(relationships=mock_relationships_2)

        # Mock the transformer to return different results for different calls
        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.side_effect = [
            [mock_graph_doc_1],
            [mock_graph_doc_2],
        ]

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        result = await prompter.aextract_triples(sample_chunks)

        assert len(result) == 2
        assert result[0].subject == "John"
        assert result[1].subject == "Python"

        # Verify transformer was called twice (once per chunk)
        assert mock_primary_transformer.convert_to_graph_documents.call_count == 2

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_retry_logic(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test retry logic when extraction fails."""
        # Mock transformer to fail first time, succeed second time
        mock_relationships = [MockRelationship("Subject", "Object", "PREDICATE")]
        mock_graph_doc = MockGraphDocument(relationships=mock_relationships)

        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.side_effect = [
            Exception("First attempt fails"),
            [mock_graph_doc],
        ]

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        result = await prompter.aextract_triples([sample_chunks[0]])

        assert len(result) == 1
        assert result[0].subject == "Subject"

        # Verify transformer was called twice (retry)
        assert mock_primary_transformer.convert_to_graph_documents.call_count == 2

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_max_retries_exceeded(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test when max retries are exceeded."""
        # Mock transformer to always fail
        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.side_effect = Exception(
            "Always fails"
        )

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        result = await prompter.aextract_triples([sample_chunks[0]])

        # Should return empty list when all retries fail
        assert len(result) == 0

        # Verify transformer was called max_retries times
        assert (
            mock_primary_transformer.convert_to_graph_documents.call_count
            == sample_config.max_retries
        )

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_empty_input(
        self, mock_transformer, mock_openai, sample_config
    ):
        """Test extracting triples from empty input."""
        prompter = GraphPrompter(sample_config)

        result = await prompter.aextract_triples([])

        assert result == []

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_extract_triples_no_relationships_found(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test when no relationships are found in the text."""
        # Mock empty graph document
        mock_graph_doc = MockGraphDocument(relationships=[])

        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.return_value = [
            mock_graph_doc
        ]

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        result = await prompter.aextract_triples([sample_chunks[0]])

        assert len(result) == 0

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    def test_extract_triples_sync_wrapper(
        self, mock_transformer, mock_openai, sample_config, sample_chunks
    ):
        """Test synchronous wrapper for extract_triples."""
        # Setup mock
        mock_relationships = [MockRelationship("Test", "Entity", "RELATES")]
        mock_graph_doc = MockGraphDocument(relationships=mock_relationships)

        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.return_value = [
            mock_graph_doc
        ]

        prompter = GraphPrompter(sample_config)
        prompter._primary_transformer = mock_primary_transformer

        # Test sync method (should work with asyncio.run)
        result = prompter.extract_triples([sample_chunks[0]])

        assert len(result) == 1
        assert result[0].subject == "Test"

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    def test_get_stats(self, mock_transformer, mock_openai, sample_config):
        """Test getting statistics about the GraphPrompter."""
        prompter = GraphPrompter(sample_config)

        stats = prompter.get_stats()

        expected_stats = {
            "primary_model": "gpt-3.5-turbo",
            "fallback_model": "gpt-4",
            "primary_max_tokens": 100,
            "fallback_max_tokens": 500,
            "batch_size": 2,
            "strict_mode": True,
        }

        assert stats == expected_stats

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_chunk_exceeds_fallback_limit(
        self, mock_transformer, mock_openai, sample_config
    ):
        """Test handling chunks that exceed even the fallback token limit."""
        # Create a chunk that exceeds fallback limit
        huge_chunk = Chunk.create(
            text="Test", start_idx=0, end_idx=4, chunk_id="huge_chunk"
        )
        huge_chunk.token_count = 1000  # Exceeds fallback_max_tokens (500)

        prompter = GraphPrompter(sample_config)

        result = await prompter.aextract_triples([huge_chunk])

        # Should return empty list for chunks that are too large
        assert len(result) == 0


class TestIntegration:
    """Integration tests for the complete graph extraction pipeline."""

    @patch("src.tubeatlas.rag.graph_extraction.prompter.ChatOpenAI")
    @patch("src.tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer")
    async def test_realistic_transcript_extraction(self, mock_transformer, mock_openai):
        """Test with realistic transcript data."""
        # Create realistic chunks
        chunks = [
            Chunk.create(
                text="Hi everyone, I'm John and I teach machine learning at Stanford. Today we'll discuss neural networks.",
                start_idx=0,
                end_idx=100,
                chunk_id="intro",
                metadata={
                    "speaker": "John",
                    "timestamp": "00:01:00",
                    "topic": "introduction",
                },
            ),
            Chunk.create(
                text="Neural networks are inspired by biological neurons. They consist of layers that process information.",
                start_idx=101,
                end_idx=200,
                chunk_id="explanation",
                metadata={
                    "speaker": "John",
                    "timestamp": "00:02:30",
                    "topic": "neural_networks",
                },
            ),
        ]

        # Mock realistic relationships
        mock_relationships_1 = [
            MockRelationship("John", "machine learning", "TEACHES"),
            MockRelationship("John", "Stanford", "WORKS_AT"),
        ]
        mock_relationships_2 = [
            MockRelationship("neural networks", "biological neurons", "INSPIRED_BY"),
            MockRelationship("neural networks", "layers", "CONSIST_OF"),
        ]

        mock_graph_doc_1 = MockGraphDocument(relationships=mock_relationships_1)
        mock_graph_doc_2 = MockGraphDocument(relationships=mock_relationships_2)

        mock_primary_transformer = Mock()
        mock_primary_transformer.convert_to_graph_documents.side_effect = [
            [mock_graph_doc_1],
            [mock_graph_doc_2],
        ]

        config = GraphPrompterConfig(batch_size=1)  # Process one chunk at a time
        prompter = GraphPrompter(config)
        prompter._primary_transformer = mock_primary_transformer

        result = await prompter.aextract_triples(chunks)

        # Verify we got the expected triples
        assert len(result) == 4

        # Check provenance is properly attached
        for triple in result:
            assert "chunk_id" in triple.provenance
            assert "speaker" in triple.provenance
            assert "timestamp" in triple.provenance
            assert triple.provenance["speaker"] == "John"

        # Check specific relationships
        subjects = [t.subject for t in result]
        assert "John" in subjects
        assert "neural networks" in subjects
