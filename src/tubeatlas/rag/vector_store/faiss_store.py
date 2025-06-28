# flake8: noqa
# mypy: ignore-errors

"""
FAISS-based vector store implementation.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..chunking.base import Chunk
from .base import VectorStoreInterface

try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStoreInterface):
    """
    FAISS-based vector store with metadata filtering and persistence.

    Features:
    - Efficient similarity search using FAISS
    - Metadata filtering support
    - Persistence to disk
    - Incremental updates
    - Memory usage monitoring
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "cosine",
        normalize_embeddings: bool = True,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension (inferred from first batch if None)
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2", "ip")
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu and numpy are required for FaissVectorStore. "
                "Install with: pip install faiss-cpu numpy"
            )

        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.normalize_embeddings = normalize_embeddings

        # Validate parameters
        if self.index_type not in ["flat", "ivf", "hnsw"]:
            raise ValueError(f"Unsupported index_type: {index_type}")
        if self.metric not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Unsupported metric: {metric}")

        # Initialize storage
        self.index = None
        self.chunks = {}  # chunk_id -> Chunk
        self.id_to_index = {}  # chunk_id -> faiss_index
        self.index_to_id = {}  # faiss_index -> chunk_id
        self._next_index = 0

        logger.info(
            f"Initialized FaissVectorStore with {index_type} index and {metric} metric"
        )

    def build_index(
        self, chunks: List[Chunk], embeddings: List[List[float]], **kwargs
    ) -> None:
        """
        Build the FAISS index from chunks and embeddings.

        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors
            **kwargs: Additional parameters (nlist for IVF, M for HNSW)
        """
        self.validate_chunks_and_embeddings(chunks, embeddings)

        if not chunks:
            logger.warning("Building index with empty chunks list")
            return

        # Infer dimension if not set
        if self.dimension is None:
            self.dimension = len(embeddings[0])
            logger.info(f"Inferred embedding dimension: {self.dimension}")

        # Clear existing data
        self.clear()

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize if using cosine similarity
        if self.normalize_embeddings and self.metric == "cosine":
            embeddings_array = self._normalize_embeddings(embeddings_array)

        # Create FAISS index
        self.index = self._create_faiss_index(**kwargs)

        # Add embeddings to index
        self.index.add(embeddings_array)

        # Store chunks and mappings
        for i, chunk in enumerate(chunks):
            faiss_idx = self._next_index
            self.chunks[chunk.id] = chunk
            self.id_to_index[chunk.id] = faiss_idx
            self.index_to_id[faiss_idx] = chunk.id
            self._next_index += 1

        logger.info(f"Built FAISS index with {len(chunks)} chunks")

    def add(self, chunks: List[Chunk], embeddings: List[List[float]], **kwargs) -> None:
        """
        Add new chunks and embeddings to existing index.

        Args:
            chunks: List of new Chunk objects
            embeddings: List of embedding vectors for new chunks
            **kwargs: Additional parameters
        """
        self.validate_chunks_and_embeddings(chunks, embeddings)

        if not chunks:
            return

        # Initialize index if it doesn't exist
        if self.index is None:
            self.build_index(chunks, embeddings, **kwargs)
            return

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize if using cosine similarity
        if self.normalize_embeddings and self.metric == "cosine":
            embeddings_array = self._normalize_embeddings(embeddings_array)

        # Add to FAISS index
        self.index.add(embeddings_array)

        # Store chunks and update mappings
        for i, chunk in enumerate(chunks):
            if chunk.id in self.chunks:
                logger.warning(f"Chunk {chunk.id} already exists, overwriting")
                # Remove old mapping
                old_faiss_idx = self.id_to_index[chunk.id]
                del self.index_to_id[old_faiss_idx]

            faiss_idx = self._next_index
            self.chunks[chunk.id] = chunk
            self.id_to_index[chunk.id] = faiss_idx
            self.index_to_id[faiss_idx] = chunk.id
            self._next_index += 1

        logger.info(f"Added {len(chunks)} chunks to FAISS index")

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} "
                f"doesn't match index dimension {self.dimension}"
            )

        # Convert to numpy array and normalize if needed
        query_array = np.array([query_embedding], dtype=np.float32)
        if self.normalize_embeddings and self.metric == "cosine":
            query_array = self._normalize_embeddings(query_array)

        # Search in FAISS
        search_k = min(k * 2, len(self.chunks))  # Get more results for filtering
        distances, indices = self.index.search(query_array, search_k)

        # Convert results to chunks with scores
        results = []
        for distance, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:  # FAISS returns -1 for invalid indices
                continue

            chunk_id = self.index_to_id.get(faiss_idx)
            if chunk_id is None:
                continue

            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue

            # Apply metadata filters if provided
            if filters and not self._matches_filters(chunk, filters):
                continue

            # Convert distance to similarity score
            similarity = self._distance_to_similarity(distance)
            results.append((chunk, similarity))

            # Stop when we have enough results
            if len(results) >= k:
                break

        return results

    def get_by_ids(self, chunk_ids: List[str]) -> List[Optional[Chunk]]:
        """Retrieve chunks by their IDs."""
        return [self.chunks.get(chunk_id) for chunk_id in chunk_ids]

    def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.

        Note: FAISS doesn't support efficient deletion, so this marks
        chunks as deleted but doesn't remove them from the index.
        A rebuild is needed to actually remove deleted chunks.
        """
        deleted_count = 0

        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                # Remove from our mappings
                faiss_idx = self.id_to_index.get(chunk_id)
                if faiss_idx is not None:
                    del self.id_to_index[chunk_id]
                    del self.index_to_id[faiss_idx]

                del self.chunks[chunk_id]
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} chunks (index rebuild recommended)")

        return deleted_count

    def persist(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize_embeddings": self.normalize_embeddings,
            "next_index": self._next_index,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Persisted vector store to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load the vector store from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vector store path {path} does not exist")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric = metadata["metric"]
        self.normalize_embeddings = metadata["normalize_embeddings"]
        self._next_index = metadata["next_index"]
        self.id_to_index = metadata["id_to_index"]
        self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}

        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = None

        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"Loaded vector store from {path} with {len(self.chunks)} chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "chunk_count": len(self.chunks),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize_embeddings": self.normalize_embeddings,
        }

        if self.index is not None:
            stats["faiss_index_size"] = self.index.ntotal
            stats["is_trained"] = getattr(self.index, "is_trained", True)

        return stats

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = None
        self.chunks.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self._next_index = 0
        logger.info("Cleared vector store")

    def _create_faiss_index(self, **kwargs) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        if self.dimension is None:
            raise ValueError("Dimension must be set before creating index")

        if self.index_type == "flat":
            if self.metric == "cosine" or self.metric == "ip":
                index = faiss.IndexFlatIP(self.dimension)
            else:  # l2
                index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "ivf":
            nlist = kwargs.get("nlist", 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            if self.metric == "cosine" or self.metric == "ip":
                index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_L2
                )

        elif self.index_type == "hnsw":
            M = kwargs.get("M", 16)  # Number of connections
            index = faiss.IndexHNSWFlat(self.dimension, M)
            if self.metric == "cosine" or self.metric == "ip":
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                index.metric_type = faiss.METRIC_L2

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return index

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score."""
        if self.metric == "cosine" or self.metric == "ip":
            # For inner product, higher is better (already similarity)
            return float(distance)
        else:  # l2
            # For L2 distance, lower is better, convert to similarity
            return 1.0 / (1.0 + float(distance))

    def _matches_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk metadata matches the provided filters."""
        if not chunk.metadata:
            return not filters  # If no metadata, only match if no filters

        for key, expected_value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if isinstance(expected_value, list):
                # If expected value is a list, check if chunk value is in it
                if chunk_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Support range queries like {"$gte": 10, "$lte": 20}
                if "$gte" in expected_value and (
                    chunk_value is None or chunk_value < expected_value["$gte"]
                ):
                    return False
                if "$lte" in expected_value and (
                    chunk_value is None or chunk_value > expected_value["$lte"]
                ):
                    return False
                if "$gt" in expected_value and (
                    chunk_value is None or chunk_value <= expected_value["$gt"]
                ):
                    return False
                if "$lt" in expected_value and (
                    chunk_value is None or chunk_value >= expected_value["$lt"]
                ):
                    return False
            else:
                # Exact match
                if chunk_value != expected_value:
                    return False

        return True
