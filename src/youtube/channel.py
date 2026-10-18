"""Channel fetching functionality."""

from typing import Any

from googleapiclient.errors import HttpError

from src.core.exceptions import YouTubeAPIError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ChannelFetcher:
    """Handles fetching YouTube channel information."""

    def __init__(self, youtube_client: Any) -> None:
        """Initialize the channel fetcher.

        Args:
            youtube_client: Google API YouTube client instance
        """
        self.youtube = youtube_client

    def get_channel_id(self, channel_handle: str) -> str:
        """Get channel ID from channel handle.

        Args:
            channel_handle: YouTube channel handle (e.g., @channelname)

        Returns:
            Channel ID

        Raises:
            YouTubeAPIError: If channel not found or API error occurs
        """
        try:
            # Remove @ if present
            handle = channel_handle.lstrip("@")

            request = self.youtube.channels().list(part="id", forHandle=handle)
            response = request.execute()

            if response.get("items"):
                channel_id = response["items"][0]["id"]
                logger.info(
                    f"Found channel ID {channel_id} for handle {channel_handle}"
                )
                return channel_id

            # Try to search for the channel
            search_request = self.youtube.search().list(
                part="id", q=channel_handle, type="channel", maxResults=1
            )
            search_response = search_request.execute()

            if search_response.get("items"):
                channel_id = search_response["items"][0]["id"]["channelId"]
                logger.info(
                    f"Found channel ID {channel_id} for handle {channel_handle}"
                )
                return channel_id

            raise YouTubeAPIError(f"Channel not found: {channel_handle}")

        except HttpError as e:
            logger.error(f"YouTube API error while fetching channel ID: {e}")
            raise YouTubeAPIError(f"Failed to fetch channel ID: {e}")

    def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        """Get detailed channel information.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dictionary containing channel information

        Raises:
            YouTubeAPIError: If API error occurs
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,contentDetails,statistics", id=channel_id
            )
            response = request.execute()

            if not response.get("items"):
                raise YouTubeAPIError(f"Channel not found: {channel_id}")

            channel_data = response["items"][0]
            logger.info(f"Fetched info for channel: {channel_data['snippet']['title']}")

            return {
                "id": channel_data["id"],
                "title": channel_data["snippet"]["title"],
                "description": channel_data["snippet"]["description"],
                "subscriber_count": channel_data["statistics"].get(
                    "subscriberCount", "0"
                ),
                "video_count": channel_data["statistics"].get("videoCount", "0"),
                "view_count": channel_data["statistics"].get("viewCount", "0"),
                "uploads_playlist_id": channel_data["contentDetails"][
                    "relatedPlaylists"
                ]["uploads"],
            }

        except HttpError as e:
            logger.error(f"YouTube API error while fetching channel info: {e}")
            raise YouTubeAPIError(f"Failed to fetch channel info: {e}")
