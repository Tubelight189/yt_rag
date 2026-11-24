"""Chat service for querying indexed content."""

from typing import Any
import json
import re
from pathlib import Path

from src.core.config import Settings
from src.core.logging import get_logger
from src.rag.embeddings import get_cached_embedder
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
        embedding_provider = self.settings.resolved_embedding_provider()
        self.embedder = get_cached_embedder(
            embedding_provider == "local",
            self.settings.embedding_model,
            self.settings.openai_api_key,
        )

        # Initialize vector store
        collection_name = "youtube_rag" if embedding_provider == "local" else "youtube_rag_openai"
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=self.settings.chroma_db_path,
            expected_dimension=384 if embedding_provider == "local" else 1536,
        )

        # Initialize retriever
        self.retriever = SimpleRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.channel_registry_path = Path(self.settings.chroma_db_path).parent / "youtube_channels.json"

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

            channel = None
            match = re.search(r"(?<!\w)(@[A-Za-z0-9._-]+)", question)
            if match:
                try:
                    channels = json.loads(self.channel_registry_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    channels = {}
                handle = match.group(1).lower()
                channel = next((c for c in channels.values() if c.get("handle", "").lower() == handle), None)
                if channel:
                    where = {"channel_id": channel["channel_id"]}
                    documents = self.retriever.retrieve(question, k=k, where=where)
                    profile = self.vector_store.search(self.embedder.embed_text("overall channel profile"), k=2,
                                                       where={"source_id": f"youtube_channel:{channel['channel_id']}"})
                    documents += [{"text": t, "score": s, "metadata": m} for t, s, m in profile]
                else:
                    return {"answer": f"{match.group(1)} is not indexed yet. Index the channel first.", "sources": []}
            else:
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
