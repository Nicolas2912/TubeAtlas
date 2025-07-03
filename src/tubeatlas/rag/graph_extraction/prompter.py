"""
GraphPrompter module for extracting knowledge graph triples from transcript chunks.

This module provides a wrapper around LangChain's LLMGraphTransformer with enhanced
features like fallback model support, token counting, batch processing, and provenance tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore
from langchain_core.documents import Document
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from ..chunking.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """
    Represents a knowledge graph triple with optional metadata.

    Attributes:
        subject: The subject entity
        predicate: The relationship type
        object: The object entity
        confidence: Optional confidence score (0.0-1.0)
        provenance: Optional metadata about the source
    """

    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    provenance: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary format."""
        result = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.provenance:
            result["provenance"] = self.provenance
        return result


@dataclass
class GraphPrompterConfig:
    """Configuration for GraphPrompter."""

    primary_model: str = "gpt-3.5-turbo"
    fallback_model: str = "gpt-4"
    primary_max_tokens: int = 4000
    fallback_max_tokens: int = 8000
    temperature: float = 0.0
    strict_mode: bool = True
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_instructions: str = ""

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "GraphPrompterConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        graph_config = config_data.get("graph_prompter", {})
        return cls(**graph_config)  # type: ignore


class GraphPrompter:
    """
    A wrapper around LangChain's LLMGraphTransformer with enhanced features.

    Features:
    - Primary/fallback model support based on token limits
    - Batch processing for efficiency
    - Provenance tracking from chunk metadata
    - Retry logic with exponential backoff
    - Both sync and async interfaces
    """

    def __init__(self, config: Optional[GraphPrompterConfig] = None):
        """
        Initialize GraphPrompter.

        Args:
            config: Optional configuration. If None, loads from default config file.
        """
        self.config = config or self._load_default_config()
        self._primary_llm = None
        self._fallback_llm = None
        self._primary_transformer = None
        self._fallback_transformer = None
        self._initialize_components()

    def _load_default_config(self) -> GraphPrompterConfig:
        """Load default configuration from YAML file."""
        config_path = Path(__file__).parent / "graph_extraction.yaml"
        if config_path.exists():
            return GraphPrompterConfig.from_yaml(config_path)
        return GraphPrompterConfig()

    def _initialize_components(self) -> None:
        """Initialize LLM and transformer components."""
        logger.info(
            f"Initializing GraphPrompter with models: {self.config.primary_model} -> {self.config.fallback_model}"
        )

        # Initialize primary components
        self._primary_llm = ChatOpenAI(
            model=self.config.primary_model, temperature=self.config.temperature
        )
        self._primary_transformer = LLMGraphTransformer(
            llm=self._primary_llm,
            strict_mode=self.config.strict_mode,
            additional_instructions=self.config.additional_instructions,
        )

        # Initialize fallback components
        self._fallback_llm = ChatOpenAI(
            model=self.config.fallback_model, temperature=self.config.temperature
        )
        self._fallback_transformer = LLMGraphTransformer(
            llm=self._fallback_llm,
            strict_mode=self.config.strict_mode,
            additional_instructions=self.config.additional_instructions,
        )

    def extract_triples(self, chunks: List[Chunk]) -> List[Triple]:
        """
        Extract triples from chunks synchronously.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted triples with provenance
        """
        return asyncio.run(self.aextract_triples(chunks))

    async def aextract_triples(self, chunks: List[Chunk]) -> List[Triple]:
        """
        Extract triples from chunks asynchronously.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted triples with provenance
        """
        if not chunks:
            return []

        logger.info(f"Extracting triples from {len(chunks)} chunks")
        all_triples = []

        # Process chunks in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i : i + self.config.batch_size]
            logger.debug(
                f"Processing batch {i//self.config.batch_size + 1}/{(len(chunks) + self.config.batch_size - 1)//self.config.batch_size}"
            )

            batch_triples = await self._process_batch(batch)
            all_triples.extend(batch_triples)

        logger.info(f"Extracted {len(all_triples)} total triples")
        return all_triples

    async def _process_batch(self, chunks: List[Chunk]) -> List[Triple]:
        """Process a batch of chunks."""
        triples = []

        for chunk in chunks:
            chunk_triples = await self._extract_from_chunk(chunk)
            triples.extend(chunk_triples)

        return triples

    async def _extract_from_chunk(self, chunk: Chunk) -> List[Triple]:
        """Extract triples from a single chunk with fallback logic."""
        # Determine which model to use based on token count
        use_fallback = chunk.token_count > self.config.primary_max_tokens

        if use_fallback:
            logger.info(
                f"Using fallback model for chunk {chunk.id} ({chunk.token_count} tokens)"
            )
            transformer = self._fallback_transformer
            model_name = self.config.fallback_model

            # Check if even fallback model can handle this
            if chunk.token_count > self.config.fallback_max_tokens:
                logger.warning(
                    f"Chunk {chunk.id} exceeds fallback token limit ({chunk.token_count} > {self.config.fallback_max_tokens})"
                )
                return []
        else:
            logger.debug(
                f"Using primary model for chunk {chunk.id} ({chunk.token_count} tokens)"
            )
            transformer = self._primary_transformer
            model_name = self.config.primary_model

        # Extract triples with retry logic
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                triples = await self._extract_with_transformer(transformer, chunk)
                end_time = time.time()

                logger.debug(
                    f"Extracted {len(triples)} triples from chunk {chunk.id} using {model_name} (took {end_time - start_time:.2f}s)"
                )
                return triples

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed for chunk {chunk.id}: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                else:
                    logger.error(f"All attempts failed for chunk {chunk.id}")
                    return []

        # This should never be reached, but add for type safety
        return []

    async def _extract_with_transformer(
        self, transformer: LLMGraphTransformer, chunk: Chunk
    ) -> List[Triple]:
        """Extract triples using the specified transformer."""
        # Create document from chunk
        doc = Document(page_content=chunk.text)

        # Convert to graph documents
        graph_documents = await asyncio.get_event_loop().run_in_executor(
            None, transformer.convert_to_graph_documents, [doc]
        )

        if not graph_documents:
            logger.warning(f"No graph documents generated for chunk {chunk.id}")
            return []

        graph_doc = graph_documents[0]
        triples = []

        # Convert relationships to triples
        for rel in graph_doc.relationships:
            # Format predicate (lowercase, replace underscores with spaces)
            predicate_formatted = rel.type.lower().replace("_", " ")

            triple = Triple(
                subject=rel.source.id,
                predicate=predicate_formatted,
                object=rel.target.id,
                provenance={
                    "chunk_id": chunk.id,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "token_count": chunk.token_count,
                    **chunk.metadata,
                },
            )
            triples.append(triple)

        return triples

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the GraphPrompter configuration."""
        return {
            "primary_model": self.config.primary_model,
            "fallback_model": self.config.fallback_model,
            "primary_max_tokens": self.config.primary_max_tokens,
            "fallback_max_tokens": self.config.fallback_max_tokens,
            "batch_size": self.config.batch_size,
            "strict_mode": self.config.strict_mode,
        }
