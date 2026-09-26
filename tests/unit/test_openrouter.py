"""Verify that production AI clients route requests through OpenRouter."""

from unittest.mock import MagicMock, patch

import pytest

from tubeatlas.config.settings import Settings, settings
from tubeatlas.rag.embedding.openai import OpenAIEmbedder
from tubeatlas.rag.graph_extraction.prompter import GraphPrompter


def test_settings_load_openrouter_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENROUTER_API_KEY=test-router-key\n")
    config = Settings(_env_file=path)
    assert config.openrouter_api_key == "test-router-key"


@pytest.mark.parametrize("model", [None, "text-embedding-3-small"])
def test_embeddings_use_openrouter_credentials_endpoint_and_model(monkeypatch, model):
    monkeypatch.setattr(settings, "openrouter_api_key", "test-router-key")
    monkeypatch.setattr(
        settings, "openrouter_embedding_model", "openai/text-embedding-3-small"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    with patch("tubeatlas.rag.embedding.openai.openai.OpenAI") as client:
        item = MagicMock(index=0, embedding=[0.1] * 1536)
        client.return_value.embeddings.create.return_value.data = [item]
        embedder = OpenAIEmbedder(model=model)
        assert len(embedder.embed_text("hello")) == 1536
        client.assert_called_once_with(
            api_key="test-router-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,
        )
        client.return_value.embeddings.create.assert_called_once_with(
            model="openai/text-embedding-3-small", input=["hello"]
        )


def test_graph_clients_use_openrouter_and_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "test-router-key")
    monkeypatch.setattr(settings, "openrouter_chat_model", "provider/custom-model")
    with (
        patch("tubeatlas.rag.graph_extraction.prompter.ChatOpenAI") as client,
        patch("tubeatlas.rag.graph_extraction.prompter.LLMGraphTransformer"),
    ):
        GraphPrompter()
        assert client.call_count == 2
        for call in client.call_args_list:
            assert call.kwargs["api_key"] == "test-router-key"
            assert call.kwargs["base_url"] == "https://openrouter.ai/api/v1"
            assert call.kwargs["model"] == "provider/custom-model"
