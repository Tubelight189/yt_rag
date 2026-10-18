"""Text chunking implementations."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.base import BaseChunker

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):
    """Recursive character-based text chunking."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Size of each chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks using recursive character splitting.

        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        chunks = self.splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        return [
            {"text": chunk, "metadata": metadata or {}}
            for chunk in chunks
        ]


class SentenceChunker(BaseChunker):
    """Sentence-based text chunking using sentence transformers."""

    def __init__(
        self,
        chunk_overlap: int = 0,
        tokens_per_chunk: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the sentence chunker.

        Args:
            chunk_overlap: Number of sentences to overlap
            tokens_per_chunk: Target tokens per chunk (default from settings)
            model_name: Sentence transformer model name
        """
        settings = get_settings()
        self.tokens_per_chunk = tokens_per_chunk or settings.chunk_size
        self.model_name = model_name or settings.embedding_model

        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
            model_name=self.model_name,
        )

    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks using sentence-based splitting.

        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        chunks = self.splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} sentence-based chunks")

        return [
            {"text": chunk, "metadata": metadata or {}}
            for chunk in chunks
        ]
