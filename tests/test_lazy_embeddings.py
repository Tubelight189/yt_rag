"""Tests for deferred local embedding model construction."""

import unittest
from unittest.mock import Mock, patch

from src.rag.embeddings import LazyLocalEmbedder, get_cached_embedder


class LazyEmbeddingTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_cached_embedder.cache_clear()

    def test_cached_local_embedder_loads_model_on_first_use_only(self) -> None:
        fake = Mock()
        fake.embed_text.return_value = [0.1]
        with patch("src.rag.embeddings.LocalEmbedder", return_value=fake) as constructor:
            embedder = get_cached_embedder(True, "test-model")

            self.assertIsInstance(embedder, LazyLocalEmbedder)
            constructor.assert_not_called()
            self.assertEqual(embedder.embed_text("question"), [0.1])
            self.assertEqual(embedder.embed_text("again"), [0.1])
            constructor.assert_called_once_with(model_name="test-model")

    def test_same_cached_proxy_is_shared(self) -> None:
        first = get_cached_embedder(True, "test-model")
        second = get_cached_embedder(True, "test-model")
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
