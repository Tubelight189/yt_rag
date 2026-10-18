"""Fetch public YouTube captions without relying on browser cookies."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import yt_dlp
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as transcript_errors

from src.core.config import Settings, get_settings
from src.core.exceptions import TranscriptNotAvailableError
from src.core.logging import get_logger

logger = get_logger(__name__)


class TranscriptProvider(Protocol):
    """A source of transcript text that does not affect indexing or RAG code."""

    name: str

    def fetch(self, video_id: str, languages: list[str]) -> str:
        """Return transcript text or raise an exception explaining why it cannot."""


class YouTubeCaptionsApiProvider:
    """Official API provider for captions on videos the OAuth user can edit."""

    name = "YouTube Captions API"

    def __init__(self, token_file: Path) -> None:
        self.token_file = token_file

    def fetch(self, video_id: str, languages: list[str]) -> str:
        try:
            credentials = Credentials.from_authorized_user_file(
                str(self.token_file), scopes=["https://www.googleapis.com/auth/youtube.force-ssl"]
            )
            youtube = build("youtube", "v3", credentials=credentials)
            tracks = youtube.captions().list(part="snippet", videoId=video_id).execute()
            items = tracks.get("items", [])
            if not items:
                raise TranscriptNotAvailableError(
                    "Official Captions API found no caption tracks you can access"
                )

            track = self._select_track(items, languages)
            if not track:
                raise TranscriptNotAvailableError(
                    "Official Captions API found no caption track in the requested languages"
                )
            downloaded = youtube.captions().download(id=track["id"], tfmt="vtt").execute()
            text = self._vtt_to_text(downloaded)
            if not text:
                raise TranscriptNotAvailableError("Official Captions API returned an empty track")
            return text
        except HttpError as error:
            status = getattr(error.resp, "status", None)
            if status in (401, 403):
                raise TranscriptNotAvailableError(
                    "Official Captions API permission denied. Use OAuth for a YouTube account "
                    "that can edit this video."
                ) from error
            if status == 404:
                raise TranscriptNotAvailableError("Video or caption track was not found") from error
            raise TranscriptNotAvailableError(
                f"Official Captions API request failed (HTTP {status})"
            ) from error
        except (OSError, ValueError) as error:
            raise TranscriptNotAvailableError(
                f"Could not load YouTube OAuth token file: {error}"
            ) from error

    @staticmethod
    def _select_track(items: list[dict], languages: list[str]) -> dict | None:
        for language in languages:
            for item in items:
                snippet = item.get("snippet", {})
                if snippet.get("language") == language and snippet.get("status") == "serving":
                    return item
        return next(
            (item for item in items if item.get("snippet", {}).get("status") == "serving"),
            None,
        )

    @staticmethod
    def _vtt_to_text(downloaded: bytes | str) -> str:
        content = downloaded.decode("utf-8", errors="replace") if isinstance(downloaded, bytes) else downloaded
        lines: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "WEBVTT" or "-->" in stripped or stripped.isdigit():
                continue
            text = re.sub(r"<[^>]+>", "", stripped)
            if text:
                lines.append(text)
        return " ".join(lines)


class TranscriptFetcher:
    """Fetch a video's public captions and return them as plain text.

    ``youtube-transcript-api`` is used first because it is designed for this
    single job. yt-dlp remains a cookie-free fallback for videos where its
    caption extractor can succeed. Neither method bypasses authentication or
    CAPTCHA challenges.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.transcript_api = YouTubeTranscriptApi()
        self.providers: list[TranscriptProvider] = []
        if self.settings.youtube_oauth_token_file:
            self.providers.append(
                YouTubeCaptionsApiProvider(self.settings.youtube_oauth_token_file)
            )

    def get_transcript(self, video_id: str, languages: list[str] | None = None) -> str:
        """Return public captions for ``video_id`` or raise a clear error."""
        preferred_languages = languages or ["en"]
        failures: list[str] = []

        providers = [*getattr(self, "providers", []), _PublicTranscriptProvider(self), _YtDlpProvider(self)]
        for provider in providers:
            try:
                transcript = provider.fetch(video_id, preferred_languages)
                logger.info("Fetched transcript for video %s using %s", video_id, provider.name)
                return transcript
            except Exception as error:
                message = self._describe_error(error)
                failures.append(f"{provider.name}: {message}")
                logger.warning(
                    "%s failed for %s [%s]: %s",
                    provider.name, video_id, type(error).__name__, str(error),
                )
                logger.debug("%s traceback for %s", provider.name, video_id, exc_info=True)

        raise TranscriptNotAvailableError(
            f"Could not retrieve transcript for video {video_id}. " + "; ".join(failures)
        )

    def _get_with_transcript_api(self, video_id: str, languages: list[str]) -> str:
        fetched_transcript = self.transcript_api.fetch(video_id, languages=languages)
        text = " ".join(
            snippet.text.strip()
            for snippet in fetched_transcript
            if snippet.text and snippet.text.strip()
        )
        if not text:
            raise TranscriptNotAvailableError("YouTube returned an empty transcript")
        return text

    def _get_with_ytdlp(self, video_id: str, languages: list[str]) -> str:
        """Use yt-dlp only as a public, cookie-free caption fallback."""
        options = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": languages,
            "subtitlesformat": "json3",
            "quiet": True,
            "no_warnings": True,
        }
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=False)
            captions, language = self._select_captions(info, languages)
            json3_caption = next(
                (caption for caption in captions if caption.get("ext") == "json3"), None
            )
            if not json3_caption:
                raise TranscriptNotAvailableError(
                    f"No JSON3 captions available for language {language}"
                )
            caption_data = ydl.urlopen(json3_caption["url"]).read().decode("utf-8")

        events = json.loads(caption_data).get("events", [])
        text = " ".join(
            segment.get("utf8", "").strip()
            for event in events
            for segment in (event.get("segs") or [])
            if segment.get("utf8", "").strip()
        )
        if not text:
            raise TranscriptNotAvailableError("yt-dlp returned an empty transcript")
        return text

    @staticmethod
    def _select_captions(info: dict, languages: Iterable[str]) -> tuple[list[dict], str]:
        subtitles = info.get("subtitles", {})
        automatic_captions = info.get("automatic_captions", {})
        for language in languages:
            if subtitles.get(language):
                return subtitles[language], language
            if automatic_captions.get(language):
                return automatic_captions[language], language
        if automatic_captions.get("en"):
            return automatic_captions["en"], "en"
        raise TranscriptNotAvailableError("No captions in the requested languages")

    @staticmethod
    def _describe_error(error: Exception) -> str:
        """Turn third-party errors into an actionable, safe user message."""
        name = type(error).__name__
        detail = " ".join(str(error).split())[:500]
        lower_detail = detail.lower()
        if isinstance(error, (transcript_errors.NoTranscriptFound,
                              transcript_errors.TranscriptsDisabled,
                              transcript_errors.VideoUnavailable)):
            return f"captions are unavailable ({name})"
        if isinstance(error, (transcript_errors.RequestBlocked, transcript_errors.IpBlocked)):
            return f"YouTube blocked this automated request ({name}; likely IP/rate limit)"
        if "sign in to confirm" in lower_detail and "not a bot" in lower_detail:
            return "YouTube requires sign-in to continue (yt-dlp bot check)"
        if isinstance(error, (transcript_errors.AgeRestricted,
                              transcript_errors.VideoUnplayable,
                              transcript_errors.CookieError)):
            return f"authentication or playback restriction ({name})"
        if isinstance(error, transcript_errors.YouTubeRequestFailed) or any(
            word in lower_detail for word in ("timeout", "connection", "network")
        ):
            return f"network request failed ({name}: {detail})"
        return f"retrieval failed ({name}: {detail})"


class _PublicTranscriptProvider:
    name = "youtube-transcript-api"

    def __init__(self, fetcher: TranscriptFetcher) -> None:
        self.fetcher = fetcher

    def fetch(self, video_id: str, languages: list[str]) -> str:
        return self.fetcher._get_with_transcript_api(video_id, languages)


class _YtDlpProvider:
    name = "yt-dlp"

    def __init__(self, fetcher: TranscriptFetcher) -> None:
        self.fetcher = fetcher

    def fetch(self, video_id: str, languages: list[str]) -> str:
        return self.fetcher._get_with_ytdlp(video_id, languages)
