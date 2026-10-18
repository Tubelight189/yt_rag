"""Regression tests for the persistent Chroma collection lifecycle."""

import unittest
from unittest.mock import Mock

from src.rag.vectorstore import ChromaVectorStore


class VectorStoreLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ChromaVectorStore.__new__(ChromaVectorStore)
        self.store.collection_name = "youtube_rag"
        self.store.collection = Mock()

    def test_add_clear_add_and_search_has_only_new_documents(self) -> None:
        self.store.collection.get.side_effect = [
            {"ids": ["old-1"]},
            {
                "ids": ["new-1"],
                "documents": [["new transcript"]],
                "distances": [[0.1]],
                "metadatas": [[{"video_id": "new-video", "source_url": "https://youtu.be/new"}]],
            },
        ]
        self.store.collection.query.return_value = {
            "documents": [["new transcript"]],
            "distances": [[0.1]],
            "metadatas": [[{"video_id": "new-video", "source_url": "https://youtu.be/new"}]],
        }

        self.store.add_documents(
            ["old transcript"], [[0.1]], [{"video_id": "old-video"}], ids=["old-1"]
        )
        self.store.clear_documents()
        self.store.add_documents(
            ["new transcript"],
            [[0.2]],
            [{"video_id": "new-video", "source_url": "https://youtu.be/new"}],
            ids=["new-1"],
        )
        results = self.store.search([0.2], k=5)

        self.store.collection.delete.assert_called_once_with(ids=["old-1"])
        self.assertEqual(results, [
            ("new transcript", 0.9, {"video_id": "new-video", "source_url": "https://youtu.be/new"})
        ])
        self.assertNotIn("old-video", str(results))


if __name__ == "__main__":
    unittest.main()
