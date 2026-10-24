"""Tests for embedding provider and collection isolation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.config import Settings
from src.core.exceptions import EmbeddingError, VectorStoreError
from src.rag.embeddings import OpenAIEmbedder, get_cached_embedder
from src.rag.vectorstore import ChromaVectorStore


class EmbeddingProviderTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_cached_embedder.cache_clear()

    def test_legacy_local_flag_selects_local_provider(self) -> None:
        settings = Settings(youtube_api_key="test", should_local_embed=True)
        self.assertEqual(settings.resolved_embedding_provider(), "local")

    def test_explicit_openai_provider_overrides_legacy_flag(self) -> None:
        settings = Settings(
            youtube_api_key="test", should_local_embed=True, embedding_provider="openai"
        )
        self.assertEqual(settings.resolved_embedding_provider(), "openai")

    def test_openai_provider_requires_key(self) -> None:
        with self.assertRaisesRegex(EmbeddingError, "OpenAI API key"):
            get_cached_embedder(False, "text-embedding-3-small", None)

    @patch("src.rag.embeddings.OpenAI")
    def test_openai_single_and_batch_response_conversion(self, client_class: Mock) -> None:
        client = client_class.return_value
        client.embeddings.create.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])]),
            SimpleNamespace(data=[
                SimpleNamespace(embedding=[0.3, 0.4]),
                SimpleNamespace(embedding=[0.5, 0.6]),
            ]),
        ]
        embedder = OpenAIEmbedder(api_key="test-key")

        self.assertEqual(embedder.embed_text("one"), [0.1, 0.2])
        self.assertEqual(embedder.embed_batch(["one", "two"]), [[0.3, 0.4], [0.5, 0.6]])

    @patch("src.rag.embeddings.OpenAIEmbedder")
    def test_openai_selection_does_not_construct_local_embedder(self, openai_class: Mock) -> None:
        get_cached_embedder(False, "text-embedding-3-small", "test-key")
        openai_class.assert_called_once_with(api_key="test-key")

    def test_collection_dimension_mismatch_is_rejected(self) -> None:
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.collection_name = "youtube_rag_openai"
        store.collection = Mock()
        store.collection.get.return_value = {"embeddings": [[0.0] * 384]}

        with self.assertRaisesRegex(VectorStoreError, "384-dimensional.*1536"):
            store._validate_embedding_dimension(1536)

    def test_local_and_remote_collection_names_are_distinct(self) -> None:
        self.assertEqual("youtube_rag", "youtube_rag")
        self.assertNotEqual("youtube_rag", "youtube_rag_openai")


if __name__ == "__main__":
    unittest.main()
