"""Indexing service for processing and storing YouTube content."""

from typing import Any

from src.core.config import Settings
from src.core.exceptions import IndexingError, TranscriptNotAvailableError
from src.core.logging import get_logger
from src.rag.chunker import RecursiveChunker
from src.rag.embeddings import get_cached_embedder
from src.rag.vectorstore import ChromaVectorStore
from src.services.base import BaseService
from src.youtube.client import YouTubeClient
from src.youtube.videos import parse_video_id
from src.sources import extract_pdf, fetch_website, stable_source_id, normalize_url
from hashlib import sha256
import json
from pathlib import Path
from src.rag.generator import GeminiGenerator
from src.youtube.channel import normalize_channel_handle

logger = get_logger(__name__)


class IndexingService(BaseService):
    """Service for indexing YouTube channel content."""

    # Keep both embedding inference and each Chroma upsert bounded.  This is
    # deliberately independent of the model's internal encode batch size.
    INDEX_BATCH_SIZE = 8

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
        embedding_provider = self.settings.resolved_embedding_provider()
        self.embedder = get_cached_embedder(
            embedding_provider == "local",
            self.settings.embedding_model,
            self.settings.openai_api_key,
        )

        collection_name = "youtube_rag" if embedding_provider == "local" else "youtube_rag_openai"
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=self.settings.chroma_db_path,
            expected_dimension=384 if embedding_provider == "local" else 1536,
        )
        self.channel_registry_path = Path(self.settings.chroma_db_path).parent / "youtube_channels.json"

    def _channels(self) -> dict[str, Any]:
        try:
            return json.loads(self.channel_registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_channel(self, record: dict[str, Any]) -> None:
        self.channel_registry_path.parent.mkdir(parents=True, exist_ok=True)
        channels = self._channels()
        channels[record["channel_id"]] = record
        self.channel_registry_path.write_text(json.dumps(channels, indent=2), encoding="utf-8")

    def _generate_channel_profile(self, record: dict[str, Any], transcripts: list[str]) -> None:
        settings = getattr(self, "settings", None)
        if not transcripts or not getattr(settings, "gemini_api_key", None):
            return
        try:
            generator = GeminiGenerator(model=settings.gemini_model, api_key=settings.gemini_api_key)
            # Keep profile requests bounded while still including evidence from every indexed video.
            evidence = "\n\n".join(t[:6000] for t in transcripts)
            prompt = ("Analyze this YouTube channel as a whole. Identify its general content category, "
                      "recurring topics, themes, presentation style, and useful observations. "
                      "Base conclusions only on the indexed transcripts, say when evidence is insufficient, "
                      "and do not summarize videos separately.\n\n" + evidence)
            profile = generator.generate("Create the overall channel content profile.", [prompt])
            source_id = f"youtube_channel:{record['channel_id']}"
            metadata = {"source_id": source_id, "source_type": "youtube_channel", "workspace_id": "default",
                        "source_name": record["channel_name"], "channel_id": record["channel_id"],
                        "channel_handle": record["handle"], "channel_name": record["channel_name"]}
            chunks = self.chunker.chunk_text(profile, metadata=metadata)
            self._index_chunks(chunks, source_id=source_id, replace=True)
            record["profile"] = profile
        except Exception as error:
            logger.warning("Channel profile generation failed; video chunks remain indexed: %s", error)

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
            videos_failed = 0
            transcripts = []

            for video in videos:
                try:
                    # Get transcript
                    transcript = self.youtube_client.get_video_transcript(video["video_id"])

                    # Prepare metadata
                    metadata = {
                        "source_id": f"youtube:{video['video_id']}",
                        "source_type": "youtube",
                        "workspace_id": "default",
                        "source_name": video["title"],
                        "video_id": video["video_id"],
                        "video_title": video["title"],
                        "channel_id": channel_id,
                        "channel_handle": normalize_channel_handle(channel_handle),
                        "channel_source_id": f"youtube_channel:{channel_id}",
                        "channel_name": channel_info["title"],
                        "published_at": video["published_at"],
                    }

                    # Chunk the transcript
                    chunks = self.chunker.chunk_text(transcript, metadata=metadata)

                    self._index_chunks(
                        chunks,
                        source_id=f"youtube:{video['video_id']}",
                        replace=True,
                    )

                    total_chunks += len(chunks)
                    videos_indexed += 1
                    transcripts.append(transcript)
                    logger.info(
                        f"Indexed video: {video['title']} ({len(chunks)} chunks)"
                    )

                except TranscriptNotAvailableError as error:
                    logger.warning(
                        f"Skipping video (transcript unavailable): {video['title']} - {error}"
                    )
                    videos_skipped += 1
                    continue
                except Exception as error:
                    logger.exception("Failed to index video %s", video.get("video_id"))
                    videos_failed += 1

            record = {"channel_id": channel_id, "handle": normalize_channel_handle(channel_handle),
                      "channel_name": channel_info["title"], "description": channel_info.get("description", ""),
                      "channel_url": f"https://www.youtube.com/channel/{channel_id}",
                      "videos": [{"video_id": v["video_id"], "title": v["title"]} for v in videos],
                      "videos_discovered": len(videos), "videos_indexed": videos_indexed,
                      "videos_skipped": videos_skipped, "videos_failed": videos_failed,
                      "total_chunks": total_chunks}
            self._generate_channel_profile(record, transcripts)
            self._save_channel(record)
            stats = {
                "channel_handle": record["handle"],
                "channel_name": channel_info["title"],
                "channel_id": channel_id,
                "videos_indexed": videos_indexed,
                "videos_skipped": videos_skipped,
                "videos_failed": videos_failed,
                "videos_discovered": len(videos),
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
                "source_id": f"youtube:{video_id}", "source_type": "youtube", "workspace_id": "default", "source_name": video["title"],
                "video_id": video_id,
                "video_title": video["title"],
                "source_url": video_url.strip(),
                "channel_id": "single-video",
                "channel_name": "Single video import",
                "published_at": video.get("published_at", ""),
            }
            chunks = self.chunker.chunk_text(transcript, metadata=metadata)
            self._index_chunks(chunks, source_id=f"youtube:{video_id}", replace=True)
            return {"video_id": video_id, "video_title": video["title"], "total_chunks": len(chunks)}
        except Exception as error:
            if isinstance(error, IndexingError):
                raise
            raise IndexingError(f"Failed to index video: {error}") from error

    def _index_texts(
        self,
        texts: list[str],
        metadata: dict[str, Any],
        source_id: str,
        replace: bool = False,
    ) -> int:
        chunks = []
        for text in texts:
            chunks.extend(self.chunker.chunk_text(text, metadata=metadata))
        if not chunks:
            raise IndexingError("Source did not produce any chunks")
        self._index_chunks(chunks, source_id=source_id, replace=replace)
        return len(chunks)

    def _index_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_id: str,
        replace: bool = False,
    ) -> None:
        """Embed and persist chunks in bounded batches.

        IDs retain their source-relative ordinal, so re-indexing continues to
        upsert the same records.  The caller owns source replacement/deletion
        semantics; this helper only limits temporary batch memory.
        """
        old_ids = self.vector_store.get_source_ids(source_id) if replace else set()
        new_ids: set[str] = set()
        try:
            for start in range(0, len(chunks), self.INDEX_BATCH_SIZE):
                batch = chunks[start : start + self.INDEX_BATCH_SIZE]
                texts = [chunk["text"] for chunk in batch]
                batch_ids = [f"{source_id}:{start + offset}" for offset in range(len(batch))]
                self.vector_store.add_documents(
                    texts=texts,
                    embeddings=self.embedder.embed_batch(texts),
                    metadatas=[chunk["metadata"] for chunk in batch],
                    ids=batch_ids,
                )
                new_ids.update(batch_ids)
        except Exception:
            # Do not remove old records when a later batch fails.
            raise
        if replace:
            self.vector_store.delete_ids(sorted(old_ids - new_ids))

    def index_pdf(self, file_path: str, source_name: str | None = None) -> dict[str, Any]:
        try:
            pages = extract_pdf(file_path)
            name = source_name or file_path.split("\\")[-1]
            source_id = stable_source_id("pdf", name)
            self.vector_store.delete_source(source_id)
            count = 0
            for page in pages:
                count += self._index_texts([page["text"]], {"source_id": source_id, "source_type": "pdf", "workspace_id": "default", "source_name": name, "page_number": page["page_number"]}, source_id + f":page{page['page_number']}")
            return {"source_id": source_id, "source_name": name, "total_chunks": count}
        except Exception as error:
            raise IndexingError(f"Failed to index PDF: {error}") from error

    def index_website(self, url: str) -> dict[str, Any]:
        try:
            page = fetch_website(url)
            source_id = stable_source_id("website", page["url"])
            metadata = {"source_id": source_id, "source_type": "website", "workspace_id": "default", "source_name": page["title"], "source_url": page["url"]}
            count = self._index_texts([page["text"]], metadata, source_id, replace=True)
            return {"source_id": source_id, "source_name": page["title"], "total_chunks": count, "source_url": page["url"]}
        except Exception as error:
            raise IndexingError(f"Failed to index website: {error}") from error

    def delete_source(self, source_id: str) -> None:
        self.vector_store.delete_source(source_id)

    def list_sources(self) -> list[dict[str, Any]]:
        return self.vector_store.list_sources()

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
        video_id: str | None = None,
        video_title: str | None = None,
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
            stable_id = video_id.strip() if video_id else sha256(transcript.strip().encode()).hexdigest()[:16]
            title = video_title.strip() if video_title else "Manual Transcript"
            metadata = {
                "source_id": f"manual_transcript:{stable_id}", "source_type": "manual_transcript", "workspace_id": "default", "source_name": title,
                "video_id": stable_id,
                "video_title": title,
                "channel_id": "manual-import",
                "channel_name": channel_name.strip() or "Manual import",
                "published_at": "manual-import",
                "source_url": "",
            }
            chunks = self.chunker.chunk_text(transcript.strip(), metadata=metadata)
            if not chunks:
                raise IndexingError("Transcript did not produce any chunks")
            self._index_chunks(chunks, source_id=metadata["source_id"], replace=True)
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
