"""RAG module for chunking, embedding, vector storage, retrieval, and generation."""

from src.rag.base import (
    BaseChunker,
    BaseEmbedder,
    BaseGenerator,
    BaseRetriever,
    BaseVectorStore,
)
from src.rag.chunker import RecursiveChunker, SentenceChunker
from src.rag.embeddings import LocalEmbedder, OpenAIEmbedder
from src.rag.generator import AnthropicGenerator, GeminiGenerator, OpenAIGenerator
from src.rag.retriever import RerankingRetriever, SimpleRetriever
from src.rag.vectorstore import ChromaVectorStore

__all__ = [
    # Base classes
    "BaseChunker",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseRetriever",
    "BaseGenerator",
    # Implementations
    "RecursiveChunker",
    "SentenceChunker",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "ChromaVectorStore",
    "SimpleRetriever",
    "RerankingRetriever",
    "OpenAIGenerator",
    "AnthropicGenerator",
    "GeminiGenerator",
]
