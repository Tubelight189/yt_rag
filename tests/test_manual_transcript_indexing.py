"""Tests for the authorized manual-transcript RAG fallback."""

import unittest
from unittest.mock import Mock

from src.core.exceptions import IndexingError, TranscriptNotAvailableError
from src.services.indexing import IndexingService


class ManualTranscriptIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = IndexingService.__new__(IndexingService)
        self.service.chunker = Mock()
        self.service.embedder = Mock()
        self.service.vector_store = Mock()

    def test_indexes_text_through_chunk_embed_and_store_pipeline(self) -> None:
        self.service.chunker.chunk_text.return_value = [
            {"text": "first chunk", "metadata": {"video_id": "abc123"}},
            {"text": "second chunk", "metadata": {"video_id": "abc123"}},
        ]
        self.service.embedder.embed_batch.return_value = [[0.1], [0.2]]

        stats = self.service.index_transcript("A permitted transcript.", "abc123", "Demo video")

        self.assertEqual(stats["videos_indexed"], 1)
        self.assertEqual(stats["total_chunks"], 2)
        self.service.embedder.embed_batch.assert_called_once_with(["first chunk", "second chunk"])
        self.service.vector_store.add_documents.assert_called_once()

    def test_rejects_empty_manual_transcript(self) -> None:
        with self.assertRaisesRegex(IndexingError, "empty"):
            self.service.index_transcript("   ")

    def test_channel_index_continues_after_one_missing_transcript(self) -> None:
        self.service.youtube_client = Mock()
        self.service.youtube_client.get_channel_id.return_value = "channel-1"
        self.service.youtube_client.get_channel_info.return_value = {"title": "Demo channel"}
        self.service.youtube_client.get_channel_videos.return_value = [
            {"video_id": "missing", "title": "Missing", "published_at": "today"},
            {"video_id": "working", "title": "Working", "published_at": "today"},
        ]
        self.service.youtube_client.get_video_transcript.side_effect = [
            TranscriptNotAvailableError("captions disabled"), "usable transcript"
        ]
        self.service.chunker.chunk_text.return_value = [
            {"text": "usable transcript", "metadata": {"video_id": "working"}}
        ]
        self.service.embedder.embed_batch.return_value = [[0.1]]

        stats = self.service.index_channel("@demo", max_videos=2)

        self.assertEqual(stats["videos_indexed"], 1)
        self.assertEqual(stats["videos_skipped"], 1)
        self.assertEqual(stats["total_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
