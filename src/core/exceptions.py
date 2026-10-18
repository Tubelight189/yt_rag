"""Custom exceptions for the YouTube RAG system."""


class YouTubeRAGException(Exception):
    """Base exception for YouTube RAG system."""

    pass


class YouTubeAPIError(YouTubeRAGException):
    """Raised when YouTube API operations fail."""

    pass


class TranscriptNotAvailableError(YouTubeRAGException):
    """Raised when a video transcript is not available."""

    pass


class EmbeddingError(YouTubeRAGException):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(YouTubeRAGException):
    """Raised when vector store operations fail."""

    pass


class LLMError(YouTubeRAGException):
    """Raised when LLM operations fail."""

    pass


class IndexingError(YouTubeRAGException):
    """Raised when indexing operations fail."""

    pass
