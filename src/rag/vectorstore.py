"""Vector store implementations."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import get_settings
from src.core.exceptions import VectorStoreError
from src.core.logging import get_logger
from src.rag.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(
        self,
        collection_name: str = "youtube_rag",
        persist_directory: Path | None = None,
    ) -> None:
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database (default from settings)
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_db_path

        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Initialized ChromaDB collection '{collection_name}' at {self.persist_directory}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}")

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
        try:
            # Generate IDs if not provided
            if ids is None:
                existing_count = self.collection.count()
                ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

            # Ensure metadatas is provided
            if metadatas is None:
                metadatas = [{} for _ in texts]

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Added {len(texts)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")

    def search(
        self, query_embedding: list[float], k: int = 5, video_id: str | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of tuples (text, score, metadata)
        """
        try:
            query_args = {"query_embeddings": [query_embedding], "n_results": k}
            if video_id:
                query_args["where"] = {"video_id": video_id}
            results = self.collection.query(**query_args)

            # Format results
            documents = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            formatted_results = [
                (doc, 1 - dist, meta)  # Convert distance to similarity score
                for doc, dist, meta in zip(documents, distances, metadatas)
            ]

            logger.info(f"Retrieved {len(formatted_results)} documents from ChromaDB")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise VectorStoreError(f"Failed to search: {e}")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}")

    def clear_documents(self) -> None:
        """Remove every document without deleting the collection handle.

        Services keep collection handles for the lifetime of the app. Deleting
        the collection invalidates those handles in ChromaDB, so clearing is
        deliberately implemented as an in-place document deletion.
        """
        try:
            document_ids = self.collection.get(include=[])['ids']
            if document_ids:
                self.collection.delete(ids=document_ids)
            logger.info("Cleared %s documents from ChromaDB", len(document_ids))
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            raise VectorStoreError(f"Failed to clear documents: {e}")
