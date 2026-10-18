"""Indexing service for processing and storing YouTube content."""

from typing import Any

from src.core.config import Settings
from src.core.exceptions import IndexingError, TranscriptNotAvailableError
from src.core.logging import get_logger
from src.rag.chunker import RecursiveChunker
from src.rag.embeddings import LocalEmbedder, OpenAIEmbedder
from src.rag.vectorstore import ChromaVectorStore
from src.services.base import BaseService
from src.youtube.client import YouTubeClient
from src.youtube.videos import parse_video_id

logger = get_logger(__name__)


class IndexingService(BaseService):
    """Service for indexing YouTube channel content."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the indexing service.

        Args:
            settings: Application settings
        """
        super().__init__(settings)

        # Initialize components
        self.youtube_client = YouTubeClient(api_key=self.settings.youtube_api_key)
        self.chunker = RecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Initialize embedder based on settings
        if not self.settings.should_local_embed:
            self.embedder = OpenAIEmbedder(api_key=self.settings.openai_api_key)
        else:
            self.embedder = LocalEmbedder(model_name=self.settings.embedding_model)

        self.vector_store = ChromaVectorStore(persist_directory=self.settings.chroma_db_path)

    def index_channel(
        self, channel_handle: str, max_videos: int = 50
    ) -> dict[str, Any]:
        """Index a YouTube channel.

        Args:
            channel_handle: YouTube channel handle (e.g., @channelname)
            max_videos: Maximum number of videos to index

        Returns:
            Dictionary with indexing statistics

        Raises:
            IndexingError: If indexing fails
        """
        try:
            logger.info(f"Starting to index channel: {channel_handle}")

            # Get channel information
            channel_id = self.youtube_client.get_channel_id(channel_handle)
            channel_info = self.youtube_client.get_channel_info(channel_id)

            logger.info(f"Indexing channel: {channel_info['title']}")

            # Get videos
            videos = self.youtube_client.get_channel_videos(channel_id, max_results=max_videos)

            # Process each video
            total_chunks = 0
            videos_indexed = 0
            videos_skipped = 0

            for video in videos:
                try:
                    # Get transcript
                    transcript = self.youtube_client.get_video_transcript(video["video_id"])

                    # Prepare metadata
                    metadata = {
                        "video_id": video["video_id"],
                        "video_title": video["title"],
                        "channel_id": channel_id,
                        "channel_name": channel_info["title"],
                        "published_at": video["published_at"],
                    }

                    # Chunk the transcript
                    chunks = self.chunker.chunk_text(transcript, metadata=metadata)

                    # Generate embeddings
                    chunk_texts = [chunk["text"] for chunk in chunks]
                    embeddings = self.embedder.embed_batch(chunk_texts)

                    # Store in vector database
                    chunk_metadatas = [chunk["metadata"] for chunk in chunks]
                    self.vector_store.add_documents(
                        texts=chunk_texts,
                        embeddings=embeddings,
                        metadatas=chunk_metadatas,
                    )

                    total_chunks += len(chunks)
                    videos_indexed += 1
                    logger.info(
                        f"Indexed video: {video['title']} ({len(chunks)} chunks)"
                    )

                except TranscriptNotAvailableError as error:
                    logger.warning(
                        f"Skipping video (transcript unavailable): {video['title']} - {error}"
                    )
                    videos_skipped += 1
                    continue

            stats = {
                "channel_name": channel_info["title"],
                "channel_id": channel_id,
                "videos_indexed": videos_indexed,
                "videos_skipped": videos_skipped,
                "total_chunks": total_chunks,
            }

            logger.info(f"Indexing complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to index channel: {e}")
            raise IndexingError(f"Failed to index channel: {e}")

    def index_video(self, video_url: str) -> dict[str, Any]:
        """Index one public YouTube video's transcript."""
        try:
            video_id = parse_video_id(video_url)
            video = self.youtube_client.get_video(video_id)
            transcript = self.youtube_client.get_video_transcript(video_id)
            metadata = {
                "video_id": video_id,
                "video_title": video["title"],
                "source_url": video_url.strip(),
                "channel_id": "single-video",
                "channel_name": "Single video import",
                "published_at": video.get("published_at", ""),
            }
            chunks = self.chunker.chunk_text(transcript, metadata=metadata)
            self.vector_store.add_documents(
                texts=[chunk["text"] for chunk in chunks],
                embeddings=self.embedder.embed_batch([chunk["text"] for chunk in chunks]),
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return {"video_id": video_id, "video_title": video["title"], "total_chunks": len(chunks)}
        except Exception as error:
            if isinstance(error, IndexingError):
                raise
            raise IndexingError(f"Failed to index video: {error}") from error

    def clear_index(self) -> None:
        """Clear the entire vector store index."""
        try:
            self.vector_store.clear_documents()
            logger.info("Cleared vector store index")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise IndexingError(f"Failed to clear index: {e}")

    def index_transcript(
        self,
        transcript: str,
        video_id: str = "manual-transcript",
        video_title: str = "Manually supplied transcript",
        channel_name: str = "Manual import",
    ) -> dict[str, Any]:
        """Index transcript text supplied through an authorized user workflow.

        This deliberately does not fetch, authenticate to, or circumvent
        YouTube. It lets a demonstrator import captions they are permitted to
        use while preserving the normal chunk/embed/store RAG pipeline.
        """
        if not transcript or not transcript.strip():
            raise IndexingError("Transcript text is empty")

        try:
            metadata = {
                "video_id": video_id.strip() or "manual-transcript",
                "video_title": video_title.strip() or "Manually supplied transcript",
                "channel_id": "manual-import",
                "channel_name": channel_name.strip() or "Manual import",
                "published_at": "manual-import",
                "source_url": "",
            }
            chunks = self.chunker.chunk_text(transcript.strip(), metadata=metadata)
            if not chunks:
                raise IndexingError("Transcript did not produce any chunks")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.embed_batch(texts)
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            stats = {
                "channel_name": metadata["channel_name"],
                "channel_id": metadata["channel_id"],
                "videos_indexed": 1,
                "videos_skipped": 0,
                "total_chunks": len(chunks),
            }
            logger.info("Indexed manually supplied transcript (%s chunks)", len(chunks))
            return stats
        except IndexingError:
            raise
        except Exception as error:
            logger.error("Failed to index manually supplied transcript: %s", error)
            raise IndexingError(f"Failed to index manually supplied transcript: {error}")
