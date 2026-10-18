"""Unit tests for transcript retrieval; all external calls are mocked."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.exceptions import TranscriptNotAvailableError
from src.core.config import Settings
from src.youtube.transcripts import TranscriptFetcher
from src.youtube.transcripts import YouTubeCaptionsApiProvider
from youtube_transcript_api._errors import RequestBlocked


class TranscriptFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = TranscriptFetcher.__new__(TranscriptFetcher)
        self.fetcher.transcript_api = Mock()

    def test_returns_text_from_public_transcript_api(self) -> None:
        self.fetcher.transcript_api.fetch.return_value = [
            SimpleNamespace(text="Hello"), SimpleNamespace(text="world")
        ]
        self.assertEqual(self.fetcher.get_transcript("video123"), "Hello world")

    @patch("src.youtube.transcripts.yt_dlp.YoutubeDL")
    def test_uses_cookie_free_ytdlp_fallback(self, youtube_dl: Mock) -> None:
        self.fetcher.transcript_api.fetch.side_effect = Exception("primary unavailable")
        ydl = youtube_dl.return_value.__enter__.return_value
        ydl.extract_info.return_value = {
            "automatic_captions": {"en": [{"ext": "json3", "url": "https://captions"}]}
        }
        ydl.urlopen.return_value.read.return_value = b'{"events": [{"segs": [{"utf8": "Fallback text"}]}]}'

        self.assertEqual(self.fetcher.get_transcript("video123"), "Fallback text")
        self.assertNotIn("cookiesfrombrowser", youtube_dl.call_args.args[0])

    @patch.object(TranscriptFetcher, "_get_with_ytdlp", side_effect=Exception("not a bot"))
    def test_reports_youtube_block_after_both_methods_fail(self, _fallback: Mock) -> None:
        self.fetcher.transcript_api.fetch.side_effect = RequestBlocked("video123")
        with self.assertRaisesRegex(TranscriptNotAvailableError, "YouTube blocked"):
            self.fetcher.get_transcript("video123")

    def test_identifies_ytdlp_bot_check_with_curly_apostrophe(self) -> None:
        error = Exception("Sign in to confirm you’re not a bot")
        self.assertIn("bot check", self.fetcher._describe_error(error))

    def test_official_provider_extracts_requested_serving_vtt_track(self) -> None:
        provider = YouTubeCaptionsApiProvider.__new__(YouTubeCaptionsApiProvider)
        track = provider._select_track(
            [
                {"id": "fr", "snippet": {"language": "fr", "status": "serving"}},
                {"id": "en", "snippet": {"language": "en", "status": "serving"}},
            ],
            ["en"],
        )
        text = provider._vtt_to_text(b"WEBVTT\n\n00:00.000 --> 00:02.000\nHello <b>world</b>\n")
        self.assertEqual(track["id"], "en")
        self.assertEqual(text, "Hello world")

    def test_official_provider_reports_missing_tracks(self) -> None:
        provider = YouTubeCaptionsApiProvider.__new__(YouTubeCaptionsApiProvider)
        self.assertIsNone(provider._select_track([], ["en"]))

    def test_blank_oauth_token_setting_disables_official_provider(self) -> None:
        settings = Settings(youtube_api_key="test-key", youtube_oauth_token_file="")
        self.assertIsNone(settings.youtube_oauth_token_file)


if __name__ == "__main__":
    unittest.main()
