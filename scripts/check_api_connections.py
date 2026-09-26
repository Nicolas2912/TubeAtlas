"""Small live checks using the application's configured API clients.

Run from the repository root. Uses YouTube quota and a small amount of
OpenRouter credit; it does not process or persist transcripts.
"""

import logging
import math

from tubeatlas.rag.embedding.openai import OpenAIEmbedder
from tubeatlas.rag.graph_extraction.prompter import GraphPrompter
from tubeatlas.services.youtube_service import YouTubeService


def check_youtube():
    """Read one public video's metadata through the application client."""
    result = (
        YouTubeService()
        .youtube_client.videos()
        .list(part="snippet", id="jNQXAC9IVRw")
        .execute()
    )
    if not result.get("items"):
        raise ValueError("No metadata returned")
    return "public video metadata retrieved"


def check_chat():
    """Invoke the same chat client used by graph extraction."""
    prompter = GraphPrompter()
    response = prompter._primary_llm.invoke("Reply with exactly OK.", max_tokens=8)
    if not isinstance(response.content, str) or response.content.strip() != "OK":
        raise ValueError("Unexpected chat response")
    return f"{prompter.config.primary_model} returned OK"


def check_embeddings():
    """Generate and validate one embedding through the real embedder."""
    embedder = OpenAIEmbedder()
    vector = embedder.embed_text("TubeAtlas connection test.")
    if len(vector) != embedder.get_embedding_dimension() or not all(
        math.isfinite(value) for value in vector
    ):
        raise ValueError("Invalid embedding vector")
    return f"{embedder.api_model} returned {len(vector)} dimensions"


def main():
    # SDK exception messages can contain request URLs with API keys.
    # Report only exception types and HTTP status, never raw exceptions.
    logging.disable(logging.CRITICAL)
    failures = 0
    for name, check in (
        ("YouTube", check_youtube),
        ("OpenRouter chat", check_chat),
        ("OpenRouter embeddings", check_embeddings),
    ):
        try:
            print(f"PASS {name}: {check()}")
        except Exception as exc:
            failures += 1
            cause = exc
            if hasattr(exc, "last_attempt"):
                cause = exc.last_attempt.exception() or exc
            status = getattr(cause, "status_code", None)
            if status is None:
                status = getattr(getattr(cause, "resp", None), "status", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            print(f"FAIL {name}: {type(cause).__name__}{suffix}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
