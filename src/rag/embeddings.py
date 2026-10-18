"""Embedding generation implementations."""

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.core.exceptions import EmbeddingError
from src.core.logging import get_logger
from src.rag.base import BaseEmbedder

logger = get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding generation."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (default from settings)
        """
        settings = get_settings()
        self.model = model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise EmbeddingError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI embedder with model {model}")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")


class LocalEmbedder(BaseEmbedder):
    """Local embedding generation using sentence transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize local embedder.

        Args:
            model_name: Sentence transformer model name (default from settings)
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized local embedder with model {self.model_name}")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")
