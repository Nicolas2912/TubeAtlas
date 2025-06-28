# flake8: noqa
# mypy: ignore-errors

"""
RAG Pipeline Components

Provides end-to-end pipelines for document ingestion and retrieval.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from .chunking.base import Chunk, ChunkerInterface
from .embedding.base import EmbedderInterface
from .registry import get_registry
from .vector_store.base import VectorStoreInterface

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Pipeline for ingesting documents into the RAG system.

    Handles:
    - Text chunking using configurable strategies
    - Embedding generation with batching
    - Vector storage with metadata
    - Progress tracking and error recovery
    """

    def __init__(
        self,
        chunker_name: str = "fixed",
        embedder_name: str = "openai",
        vector_store_name: str = "faiss",
        batch_size: int = 500,
        max_memory_mb: int = 512,
        **component_kwargs,
    ):
        """
        Initialize the ingest pipeline.

        Args:
            chunker_name: Name of registered chunker to use
            embedder_name: Name of registered embedder to use
            vector_store_name: Name of registered vector store to use
            batch_size: Number of chunks to process in each batch
            max_memory_mb: Maximum memory usage before forcing batch processing
            **component_kwargs: Keyword arguments for component initialization
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb

        # Initialize components from registry
        registry = get_registry()

        chunker_kwargs = component_kwargs.get("chunker", {})
        embedder_kwargs = component_kwargs.get("embedder", {})
        vector_store_kwargs = component_kwargs.get("vector_store", {})

        # Embedder is a dependency for SemanticChunker, so create it first.
        self.embedder = registry.create_embedder(embedder_name, **embedder_kwargs)

        if chunker_name == "semantic":
            chunker_kwargs["embedder"] = self.embedder

        self.chunker = registry.create_chunker(chunker_name, **chunker_kwargs)
        self.vector_store = registry.create_vector_store(
            vector_store_name, **vector_store_kwargs
        )

        # Statistics tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "total_processing_time": 0.0,
            "errors": [],
        }

        logger.info(
            f"Initialized IngestPipeline with {chunker_name}/{embedder_name}/{vector_store_name}"
        )

    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        document_id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a list of documents.

        Args:
            documents: List of document dictionaries
            document_id_field: Field name containing document ID
            text_field: Field name containing document text
            metadata_fields: Optional list of metadata fields to preserve

        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()
        metadata_fields = metadata_fields or []

        try:
            # Process documents in batches
            all_chunks = []
            all_embeddings = []

            for i in range(0, len(documents), self.batch_size):
                batch = documents[i : i + self.batch_size]
                batch_chunks, batch_embeddings = await self._process_document_batch(
                    batch, document_id_field, text_field, metadata_fields
                )

                all_chunks.extend(batch_chunks)
                all_embeddings.extend(batch_embeddings)

                # Check memory usage and flush if needed
                estimated_memory = self._estimate_memory_usage(
                    all_chunks, all_embeddings
                )
                if estimated_memory > self.max_memory_mb:
                    await self._flush_to_vector_store(all_chunks, all_embeddings)
                    all_chunks.clear()
                    all_embeddings.clear()

            # Flush remaining chunks
            if all_chunks:
                await self._flush_to_vector_store(all_chunks, all_embeddings)

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["documents_processed"] += len(documents)

            logger.info(
                f"Ingested {len(documents)} documents in {processing_time:.2f}s"
            )

            return {
                "success": True,
                "documents_processed": len(documents),
                "processing_time": processing_time,
                "stats": self.stats.copy(),
            }

        except Exception as e:
            error_msg = f"Error during document ingestion: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return {"success": False, "error": error_msg, "stats": self.stats.copy()}

    async def stream_ingest(
        self,
        document_stream: AsyncGenerator[Dict[str, Any], None],
        document_id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Ingest documents from an async stream with back-pressure handling.

        Args:
            document_stream: Async generator yielding documents
            document_id_field: Field name containing document ID
            text_field: Field name containing document text
            metadata_fields: Optional list of metadata fields to preserve

        Yields:
            Progress updates and statistics
        """
        metadata_fields = metadata_fields or []
        batch = []
        chunks_buffer = []
        embeddings_buffer = []

        async for document in document_stream:
            try:
                batch.append(document)

                # Process batch when it reaches the configured size
                if len(batch) >= self.batch_size:
                    batch_chunks, batch_embeddings = await self._process_document_batch(
                        batch, document_id_field, text_field, metadata_fields
                    )

                    chunks_buffer.extend(batch_chunks)
                    embeddings_buffer.extend(batch_embeddings)

                    # Yield progress update
                    yield {
                        "type": "progress",
                        "documents_processed": len(batch),
                        "chunks_created": len(batch_chunks),
                        "total_documents": self.stats["documents_processed"]
                        + len(batch),
                    }

                    # Clear batch
                    batch.clear()

                    # Check memory and flush if needed
                    estimated_memory = self._estimate_memory_usage(
                        chunks_buffer, embeddings_buffer
                    )
                    if estimated_memory > self.max_memory_mb:
                        await self._flush_to_vector_store(
                            chunks_buffer, embeddings_buffer
                        )

                        yield {
                            "type": "flush",
                            "chunks_stored": len(chunks_buffer),
                            "memory_mb": estimated_memory,
                        }

                        chunks_buffer.clear()
                        embeddings_buffer.clear()

            except Exception as e:
                error_msg = f"Error processing document {document.get(document_id_field, 'unknown')}: {str(e)}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)

                yield {
                    "type": "error",
                    "error": error_msg,
                    "document_id": document.get(document_id_field),
                }

        # Process remaining documents
        if batch:
            try:
                batch_chunks, batch_embeddings = await self._process_document_batch(
                    batch, document_id_field, text_field, metadata_fields
                )
                chunks_buffer.extend(batch_chunks)
                embeddings_buffer.extend(batch_embeddings)

                yield {
                    "type": "progress",
                    "documents_processed": len(batch),
                    "chunks_created": len(batch_chunks),
                }
            except Exception as e:
                logger.error(f"Error processing final batch: {str(e)}")

        # Flush remaining chunks
        if chunks_buffer:
            await self._flush_to_vector_store(chunks_buffer, embeddings_buffer)
            yield {
                "type": "complete",
                "final_chunks_stored": len(chunks_buffer),
                "total_stats": self.stats.copy(),
            }

    async def _process_document_batch(
        self,
        documents: List[Dict[str, Any]],
        document_id_field: str,
        text_field: str,
        metadata_fields: List[str],
    ) -> Tuple[List[Chunk], List[List[float]]]:
        """Process a batch of documents into chunks and embeddings."""
        all_chunks = []

        # Chunk all documents
        for doc in documents:
            doc_id = doc.get(document_id_field)
            text = doc.get(text_field, "")

            if not text:
                logger.warning(f"Document {doc_id} has no text content")
                continue

            # Extract metadata
            metadata = {"document_id": doc_id}
            for field in metadata_fields:
                if field in doc:
                    metadata[field] = doc[field]

            # Chunk the document
            doc_chunks = self.chunker.chunk(text, metadata)

            # Add document-specific IDs
            for i, chunk in enumerate(doc_chunks):
                chunk.id = f"{doc_id}_{i}"

            all_chunks.extend(doc_chunks)

        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(chunk_texts) if chunk_texts else []

        # Update statistics
        self.stats["chunks_created"] += len(all_chunks)
        self.stats["embeddings_generated"] += len(embeddings)

        return all_chunks, embeddings

    async def _flush_to_vector_store(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        """Flush chunks and embeddings to the vector store."""
        if chunks and embeddings:
            self.vector_store.add(chunks, embeddings)
            logger.info(f"Flushed {len(chunks)} chunks to vector store")

    def _estimate_memory_usage(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> float:
        """Estimate memory usage in MB."""
        if not chunks or not embeddings:
            return 0.0

        # Rough estimation
        chunk_memory = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
        embedding_memory = len(embeddings) * len(embeddings[0]) * 4  # 4 bytes per float

        total_bytes = chunk_memory + embedding_memory
        return total_bytes / (1024 * 1024)  # Convert to MB

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "total_processing_time": 0.0,
            "errors": [],
        }


class RetrievalPipeline:
    """
    Pipeline for retrieving relevant chunks for queries.

    Handles:
    - Query embedding generation
    - Similarity search with filtering
    - Result ranking and formatting
    - Multi-document retrieval
    """

    def __init__(
        self,
        embedder_name: str = "openai",
        vector_store: Optional[VectorStoreInterface] = None,
        default_k: int = 5,
        similarity_threshold: float = 0.0,
        **embedder_kwargs,
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            embedder_name: Name of registered embedder to use
            vector_store: Pre-initialized vector store (optional)
            default_k: Default number of results to return
            similarity_threshold: Minimum similarity score for results
            **embedder_kwargs: Keyword arguments for embedder initialization
        """
        self.default_k = default_k
        self.similarity_threshold = similarity_threshold

        # Initialize embedder
        registry = get_registry()
        self.embedder = registry.create_embedder(embedder_name, **embedder_kwargs)

        # Use provided vector store or create default
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = registry.create_vector_store("faiss")

        logger.info(f"Initialized RetrievalPipeline with {embedder_name} embedder")

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        transcript_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query text
            k: Number of results to return (uses default if None)
            filters: Optional metadata filters
            transcript_ids: Optional list of transcript IDs to filter by
            **kwargs: Additional search parameters

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        k = k or self.default_k

        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)

            # Add transcript ID filter if specified
            if transcript_ids is not None:
                filters = filters or {}
                filters["transcript_id"] = transcript_ids

            # Perform similarity search
            results = self.vector_store.similarity_search(
                query_embedding, k=k, filters=filters, **kwargs
            )

            # Filter by similarity threshold
            if self.similarity_threshold > 0:
                results = [
                    (chunk, score)
                    for chunk, score in results
                    if score >= self.similarity_threshold
                ]

            logger.info(f"Retrieved {len(results)} chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []

    async def retrieve_multiple_queries(
        self,
        queries: List[str],
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, List[Tuple[Chunk, float]]]:
        """
        Retrieve results for multiple queries efficiently.

        Args:
            queries: List of query strings
            k: Number of results per query
            filters: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            Dictionary mapping queries to their results
        """
        results = {}

        # Generate embeddings for all queries
        query_embeddings = self.embedder.embed_texts(queries)

        # Retrieve for each query
        for query, embedding in zip(queries, query_embeddings):
            try:
                search_results = self.vector_store.similarity_search(
                    embedding, k=k or self.default_k, filters=filters, **kwargs
                )

                # Filter by similarity threshold
                if self.similarity_threshold > 0:
                    search_results = [
                        (chunk, score)
                        for chunk, score in search_results
                        if score >= self.similarity_threshold
                    ]

                results[query] = search_results

            except Exception as e:
                logger.error(f"Error retrieving for query '{query}': {str(e)}")
                results[query] = []

        return results

    def set_vector_store(self, vector_store: VectorStoreInterface) -> None:
        """Set or update the vector store."""
        self.vector_store = vector_store
        logger.info("Updated vector store for retrieval pipeline")

    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "embedder_class": self.embedder.__class__.__name__,
            "vector_store_class": self.vector_store.__class__.__name__,
            "default_k": self.default_k,
            "similarity_threshold": self.similarity_threshold,
        }


# Convenience functions for common use cases
async def ingest_documents_simple(
    documents: List[Dict[str, Any]],
    chunker_name: str = "fixed",
    embedder_name: str = "openai",
    vector_store_path: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple document ingestion with default settings.

    Args:
        documents: List of documents to ingest
        chunker_name: Chunker strategy to use
        embedder_name: Embedder to use
        vector_store_path: Optional path to persist vector store
        **kwargs: Additional pipeline parameters

    Returns:
        Ingestion results
    """
    pipeline = IngestPipeline(
        chunker_name=chunker_name, embedder_name=embedder_name, **kwargs
    )

    result = await pipeline.ingest_documents(documents)

    # Persist vector store if path provided
    if vector_store_path and result["success"]:
        pipeline.vector_store.persist(vector_store_path)
        result["vector_store_path"] = str(vector_store_path)

    return result


async def retrieve_simple(
    query: str,
    vector_store_path: Optional[Path] = None,
    vector_store: Optional[VectorStoreInterface] = None,
    k: int = 5,
    **kwargs,
) -> List[Tuple[Chunk, float]]:
    """
    Simple retrieval with default settings.

    Args:
        query: Search query
        vector_store_path: Path to load vector store from
        vector_store: Pre-initialized vector store
        k: Number of results to return
        **kwargs: Additional retrieval parameters

    Returns:
        List of (Chunk, similarity_score) tuples
    """
    # Initialize retrieval pipeline
    if vector_store is not None:
        pipeline = RetrievalPipeline(vector_store=vector_store, **kwargs)
    else:
        pipeline = RetrievalPipeline(**kwargs)

        # Load vector store if path provided
        if vector_store_path:
            pipeline.vector_store.load(vector_store_path)

    return await pipeline.retrieve(query, k=k)
