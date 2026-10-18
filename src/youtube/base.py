"""Base classes for YouTube API interactions."""

from abc import ABC, abstractmethod
from typing import Any


class BaseYouTubeClient(ABC):
    """Abstract base class for YouTube API clients."""

    @abstractmethod
    def __init__(self, api_key: str) -> None:
        """Initialize the YouTube client.

        Args:
            api_key: YouTube Data API v3 key
        """
        pass

    @abstractmethod
    def get_channel_id(self, channel_handle: str) -> str:
        """Get channel ID from channel handle.

        Args:
            channel_handle: YouTube channel handle (e.g., @channelname)

        Returns:
            Channel ID

        Raises:
            YouTubeAPIError: If channel not found or API error occurs
        """
        pass

    @abstractmethod
    def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        """Get channel information.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dictionary containing channel information

        Raises:
            YouTubeAPIError: If API error occurs
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_video_transcript(self, video_id: str) -> str:
        """Get transcript for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video transcript as text

        Raises:
            TranscriptNotAvailableError: If transcript is not available
        """
        pass
