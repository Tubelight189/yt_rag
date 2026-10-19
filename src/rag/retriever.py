"""Document retrieval implementations."""

from typing import Any

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.base import BaseEmbedder, BaseRetriever, BaseVectorStore

logger = get_logger(__name__)


class SimpleRetriever(BaseRetriever):
    """Simple vector similarity-based retriever."""

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        """Initialize the retriever.

        Args:
            embedder: Embedder for query encoding
            vector_store: Vector store for similarity search
        """
        
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = get_settings()

    def retrieve(self, query: str, k: int | None = None, video_id: str | None = None, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve (default from settings)

        Returns:
            List of document dictionaries with text, score, and metadata
        """
        k = k or self.settings.retrieval_k

        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k, video_id=video_id, where=where)

        # Format results
        documents = [
            {
                "text": text,
                "score": score,
                "metadata": metadata,
            }
            for text, score, metadata in results
        ]

        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents


class RerankingRetriever(BaseRetriever):
    """Retriever with reranking capability."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        initial_k: int = 20,
    ) -> None:
        """Initialize the reranking retriever.

        Args:
            embedder: Embedder for query encoding
            vector_store: Vector store for similarity search
            initial_k: Number of documents to retrieve before reranking
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.initial_k = initial_k
        self.settings = get_settings()

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve and rerank relevant documents for a query.

        Args:
            query: Search query
            k: Number of final documents to return (default from settings)

        Returns:
            List of document dictionaries with text, score, and metadata
        """
        k = k or self.settings.retrieval_k

        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Retrieve more documents than needed
        results = self.vector_store.search(query_embedding, k=self.initial_k)

        # Simple reranking based on query term overlap
        # In production, you might use a cross-encoder model here
        reranked = self._rerank_by_term_overlap(query, results)

        # Take top k after reranking
        final_results = reranked[:k]

        # Format results
        documents = [
            {
                "text": text,
                "score": score,
                "metadata": metadata,
            }
            for text, score, metadata in final_results
        ]

        logger.info(f"Retrieved and reranked {len(documents)} documents for query")
        return documents

    def _rerank_by_term_overlap(
        self, query: str, results: list[tuple[str, float, dict[str, Any]]]
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Rerank results based on term overlap with query.

        Args:
            query: Search query
            results: Initial search results

        Returns:
            Reranked results
        """
        query_terms = set(query.lower().split())

        scored_results = []
        for text, score, metadata in results:
            doc_terms = set(text.lower().split())
            overlap_score = len(query_terms.intersection(doc_terms)) / len(query_terms)

            # Combine similarity score with overlap score
            combined_score = 0.7 * score + 0.3 * overlap_score
            scored_results.append((text, combined_score, metadata))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results
