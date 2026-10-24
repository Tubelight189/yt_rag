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
        self.service.vector_store.get_source_ids.return_value = set()

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

    def test_indexes_in_bounded_batches_with_stable_ids(self) -> None:
        chunks = [
            {"text": f"chunk {i}", "metadata": {"source_id": "manual_transcript:abc123"}}
            for i in range(10)
        ]
        self.service.chunker.chunk_text.return_value = chunks
        self.service.embedder.embed_batch.side_effect = [
            [[float(i)] for i in range(8)],
            [[8.0], [9.0]],
        ]

        stats = self.service.index_transcript("A permitted transcript.", "abc123", "Demo video")

        self.assertEqual(stats["total_chunks"], 10)
        self.assertEqual(
            self.service.embedder.embed_batch.call_args_list[0].args[0],
            [f"chunk {i}" for i in range(8)],
        )
        self.assertEqual(self.service.embedder.embed_batch.call_args_list[1].args[0], ["chunk 8", "chunk 9"])
        calls = self.service.vector_store.add_documents.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["ids"], [f"manual_transcript:abc123:{i}" for i in range(8)])
        self.assertEqual(calls[1].kwargs["ids"], ["manual_transcript:abc123:8", "manual_transcript:abc123:9"])

    def test_reindexing_removes_stale_ids_only_after_success(self) -> None:
        self.service.vector_store.get_source_ids.return_value = {
            "manual_transcript:abc123:0", "manual_transcript:abc123:1", "manual_transcript:abc123:2"
        }
        self.service.chunker.chunk_text.return_value = [
            {"text": "new 0", "metadata": {}}, {"text": "new 1", "metadata": {}}
        ]
        self.service.embedder.embed_batch.return_value = [[0.1], [0.2]]

        self.service.index_transcript("updated", "abc123", "Demo video")

        self.service.vector_store.delete_ids.assert_called_once_with(["manual_transcript:abc123:2"])

    def test_failed_batch_does_not_delete_old_ids(self) -> None:
        self.service.vector_store.get_source_ids.return_value = {"manual_transcript:abc123:0"}
        self.service.chunker.chunk_text.return_value = [
            {"text": "new 0", "metadata": {}}, {"text": "new 1", "metadata": {}}
        ]
        self.service.embedder.embed_batch.side_effect = RuntimeError("embedding failed")

        with self.assertRaises(IndexingError):
            self.service.index_transcript("updated", "abc123", "Demo video")

        self.service.vector_store.delete_ids.assert_not_called()

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
