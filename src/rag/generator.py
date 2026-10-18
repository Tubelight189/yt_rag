"""Text generation implementations."""

import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

from src.core.config import get_settings
from src.core.exceptions import LLMError
from src.core.logging import get_logger
from src.rag.base import BaseGenerator

logger = get_logger(__name__)


class OpenAIGenerator(BaseGenerator):
    """OpenAI-based text generation."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        """Initialize OpenAI generator.

        Args:
            model: OpenAI model name (default from settings)
            api_key: OpenAI API key (default from settings)
        """
        settings = get_settings()
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise LLMError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI generator with model {self.model}")

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
        try:
            # Build context string
            context_str = "\n\n".join(
                [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context)]
            )

            # Default system prompt
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "Use the context to answer the question accurately. If the context doesn't contain "
                    "the information needed to answer the question, say so."
                )

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                },
            ]

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )

            answer = response.choices[0].message.content
            logger.info("Generated response using OpenAI")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMError(f"Failed to generate response: {e}")


class AnthropicGenerator(BaseGenerator):
    """Anthropic-based text generation."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str | None = None) -> None:
        """Initialize Anthropic generator.

        Args:
            model: Anthropic model name
            api_key: Anthropic API key (default from settings)
        """
        settings = get_settings()
        self.model = model
        self.api_key = api_key or settings.anthropic_api_key

        if not self.api_key:
            raise LLMError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic generator with model {self.model}")

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
        try:
            # Build context string
            context_str = "\n\n".join(
                [f"[Document {i+1}]\n{doc}" for i, doc in enumerate(context)]
            )

            # Default system prompt
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "Use the context to answer the question accurately. If the context doesn't contain "
                    "the information needed to answer the question, say so."
                )

            # Generate response
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                    }
                ],
            )

            answer = message.content[0].text
            logger.info("Generated response using Anthropic")
            return answer

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMError(f"Failed to generate response: {e}")


class GeminiGenerator(BaseGenerator):
    """Google Gemini-based text generation using the official Google GenAI SDK."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.gemini_model
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key:
            raise LLMError(
                "Gemini API key is required. Add GEMINI_API_KEY to your .env file."
            )

        self.client = genai.Client(api_key=self.api_key)
        logger.info("Initialized Gemini generator with model %s", self.model)

    def generate(
        self, query: str, context: list[str], system_prompt: str | None = None
    ) -> str:
        """Generate an answer using the same RAG context supplied to other providers."""
        context_str = "\n\n".join(
            [f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(context)]
        )
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use the context to answer the question accurately. If the context doesn't contain "
                "the information needed to answer the question, say so."
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"Context:\n{context_str}\n\nQuestion: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                ),
            )
            answer = response.text
            if not answer or not answer.strip():
                raise LLMError("Gemini returned an empty response")
            logger.info("Generated response using Gemini")
            return answer.strip()
        except LLMError:
            raise
        except Exception as error:
            logger.error("Gemini generation failed: %s", error)
            raise LLMError(
                "Gemini could not generate a response. Check GEMINI_API_KEY, "
                f"GEMINI_MODEL, quota, and network access. Details: {error}"
            )
