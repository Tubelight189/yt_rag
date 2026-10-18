"""Base classes for RAG components."""

from abc import ABC, abstractmethod
from typing import Any


class BaseChunker(ABC):
    """Abstract base class for text chunking."""

    @abstractmethod
    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Split text into chunks.

        Args:
            text: Text to split
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        pass


class BaseEmbedder(ABC):
    """Abstract base class for generating embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            texts: Document texts
            embeddings: Document embeddings
            metadatas: Optional document metadata
            ids: Optional document IDs
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of tuples (text, score, metadata)
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    def clear_documents(self) -> None:
        """Remove all documents while keeping the collection available."""
        pass


class BaseRetriever(ABC):
    """Abstract base class for document retrieval."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of document dictionaries with text and metadata
        """
        pass


class BaseGenerator(ABC):
    """Abstract base class for text generation."""

    @abstractmethod
    def generate(
        self, query: str, context: list[str], system_prompt: str | None = None
    ) -> str:
        """Generate a response based on query and context.

        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        pass
