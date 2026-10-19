"""Chat service for querying indexed content."""

from typing import Any

from src.core.config import Settings
from src.core.logging import get_logger
from src.rag.embeddings import LocalEmbedder, OpenAIEmbedder
from src.rag.generator import AnthropicGenerator, GeminiGenerator, OpenAIGenerator
from src.rag.retriever import SimpleRetriever
from src.rag.vectorstore import ChromaVectorStore
from src.services.base import BaseService

logger = get_logger(__name__)


class ChatService(BaseService):
    """Service for chatting with indexed YouTube content."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the chat service.

        Args:
            settings: Application settings
        """
        super().__init__(settings)

        # Initialize embedder
        if not self.settings.should_local_embed:
            self.embedder = OpenAIEmbedder(api_key=self.settings.openai_api_key)
        else:
            self.embedder = LocalEmbedder(model_name=self.settings.embedding_model)

        # Initialize vector store
        self.vector_store = ChromaVectorStore(persist_directory=self.settings.chroma_db_path)

        # Initialize retriever
        self.retriever = SimpleRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        # Initialize generator based on provider
        provider = self.settings.llm_provider.lower()
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI provider")
            self.generator = OpenAIGenerator(
                model=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
            )
        elif provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("Anthropic API key required for Anthropic provider")
            self.generator = AnthropicGenerator(
                model=self.settings.llm_model,
                api_key=self.settings.anthropic_api_key,
            )
        elif provider == "gemini":
            if not self.settings.gemini_api_key:
                raise ValueError(
                    "Gemini API key required. Add GEMINI_API_KEY to your .env file."
                )
            self.generator = GeminiGenerator(
                model=self.settings.gemini_model,
                api_key=self.settings.gemini_api_key,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    def ask(
        self,
        question: str,
        k: int | None = None,
        system_prompt: str | None = None,
        video_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Ask a question about the indexed content.

        Args:
            question: User question
            k: Number of documents to retrieve (default from settings)
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with answer and source documents
        """
        try:
            logger.info(f"Processing question: {question}")

            # Retrieve relevant documents
            documents = self.retriever.retrieve(question, k=k, video_id=video_id, where={"source_id": source_id} if source_id else None)

            if not documents:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                }

            # Extract context texts
            context_texts = [doc["text"] for doc in documents]

            # Generate answer
            answer = self.generator.generate(
                query=question,
                context=context_texts,
                system_prompt=system_prompt,
            )

            # Format source information
            sources = []
            for doc in documents:
                metadata = doc.get("metadata", {})
                sources.append(
                    {
                        "source_id": metadata.get("source_id", ""),
                        "source_type": metadata.get("source_type", ""),
                        "video_title": metadata.get("video_title", "Unknown"),
                        "video_id": metadata.get("video_id", ""),
                        "channel_name": metadata.get("channel_name", "Unknown"),
                        "score": doc.get("score", 0.0),
                    }
                )

            result = {
                "answer": answer,
                "sources": sources,
            }

            logger.info("Generated answer successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
            }
