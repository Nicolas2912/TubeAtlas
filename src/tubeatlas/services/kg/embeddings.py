from __future__ import annotations

"""Lightweight embedding helper for KG triple similarity.

During production we call OpenAI Embedding API. In dev / test environments,
OpenAI may be unavailable – in that case we fall back to a deterministic hash-
based pseudo-embedding that is *consistent within a runtime* and works for unit
testing / local runs that do not require semantic similarity.
"""

from hashlib import sha256
from os import getenv
from typing import List

import numpy as np  # type: ignore

try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    openai = None  # type: ignore


_DEFAULT_MODEL = "text-embedding-ada-002"


def _hash_embedding(text: str, dim: int = 128) -> np.ndarray:  # noqa: D401
    """Return a deterministic pseudo-embedding based on SHA-256 digest.

    Useful when the real embedding backend is unavailable. Not semantically
    meaningful but preserves *equality* for identical strings which is enough
    for unit tests around duplicate detection.
    """
    digest = sha256(text.encode("utf-8")).digest()
    # Repeat digest until we have enough bytes
    needed = dim * 4  # float32
    full = (digest * ((needed // len(digest)) + 1))[:needed]
    arr = np.frombuffer(full, dtype=np.uint8).astype(np.float32)
    arr = arr.reshape(-1, 4).mean(axis=1)  # aggregate to dim elements
    # Normalise to unit length to match cosine similarity expectations
    norm = np.linalg.norm(arr)
    return arr / norm if norm else arr


def embed_text(text: str, model: str = _DEFAULT_MODEL) -> np.ndarray:  # noqa: D401
    """Return a float32 embedding vector for *text*.

    If OpenAI client & API key are available, call the model; otherwise fallback
    to `_hash_embedding` so that the rest of the pipeline still works.
    """
    if openai is None or getenv("OPENAI_API_KEY") is None:  # pragma: no cover
        return _hash_embedding(text, dim=128)

    response = openai.Embedding.create(input=[text], model=model)
    vec: List[float] = response["data"][0]["embedding"]
    return np.asarray(vec, dtype=np.float32)