"""YouTube API client implementation."""

from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.core.exceptions import YouTubeAPIError
from src.core.logging import get_logger
from src.youtube.base import BaseYouTubeClient
from src.youtube.channel import ChannelFetcher
from src.youtube.transcripts import TranscriptFetcher
from src.youtube.videos import VideoFetcher

logger = get_logger(__name__)


class YouTubeClient(BaseYouTubeClient):
    """YouTube API client implementation."""

    def __init__(self, api_key: str) -> None:
        """Initialize the YouTube client.

        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key
        try:
            self.youtube = build("youtube", "v3", developerKey=api_key)
            logger.info("YouTube client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
            raise YouTubeAPIError(f"Failed to initialize YouTube client: {e}")

        # Initialize specialized fetchers
        self.channel_fetcher = ChannelFetcher(self.youtube)
        self.video_fetcher = VideoFetcher(self.youtube)
        self.transcript_fetcher = TranscriptFetcher()

    def get_channel_id(self, channel_handle: str) -> str:
        """Get channel ID from channel handle.

        Args:
            channel_handle: YouTube channel handle (e.g., @channelname)

        Returns:
            Channel ID

        Raises:
            YouTubeAPIError: If channel not found or API error occurs
        """
        return self.channel_fetcher.get_channel_id(channel_handle)

    def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        """Get channel information.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dictionary containing channel information

        Raises:
            YouTubeAPIError: If API error occurs
        """
        return self.channel_fetcher.get_channel_info(channel_id)

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Get videos from a channel.

        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch

        Returns:
            List of video information dictionaries

        Raises:
            YouTubeAPIError: If API error occurs
        """
        return self.video_fetcher.get_channel_videos(channel_id, max_results)

    def get_video_transcript(self, video_id: str) -> str:
        """Get transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video transcript as text

        Raises:
            TranscriptNotAvailableError: If transcript is not available
        """
        return self.transcript_fetcher.get_transcript(video_id)

    def get_video(self, video_id: str) -> dict[str, Any]:
        return self.video_fetcher.get_video(video_id)
