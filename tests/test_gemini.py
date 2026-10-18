"""Unit tests for the Gemini LLM provider; no network calls are made."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.config import Settings
from src.core.exceptions import LLMError
from src.rag.generator import GeminiGenerator
from src.services.chat import ChatService


class GeminiGeneratorTests(unittest.TestCase):
    @patch("src.rag.generator.get_settings")
    def test_missing_key_has_beginner_friendly_message(self, get_settings: Mock) -> None:
        get_settings.return_value = SimpleNamespace(gemini_model="gemini-test", gemini_api_key=None)
        with self.assertRaisesRegex(LLMError, "GEMINI_API_KEY"):
            GeminiGenerator()

    @patch("src.rag.generator.genai.Client")
    @patch("src.rag.generator.get_settings")
    def test_sends_rag_context_and_returns_response(
        self, get_settings: Mock, client_class: Mock
    ) -> None:
        get_settings.return_value = SimpleNamespace(
            gemini_model="gemini-test", gemini_api_key="test-key"
        )
        client_class.return_value.models.generate_content.return_value = SimpleNamespace(
            text="  Gemini answer  "
        )

        generator = GeminiGenerator()
        answer = generator.generate("What happened?", ["Transcript excerpt"])

        self.assertEqual(answer, "Gemini answer")
        kwargs = client_class.return_value.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-test")
        self.assertIn("Transcript excerpt", kwargs["contents"])
        self.assertIn("What happened?", kwargs["contents"])

    @patch("src.rag.generator.genai.Client")
    @patch("src.rag.generator.get_settings")
    def test_api_error_becomes_llm_error(self, get_settings: Mock, client_class: Mock) -> None:
        get_settings.return_value = SimpleNamespace(
            gemini_model="invalid-model", gemini_api_key="test-key"
        )
        client_class.return_value.models.generate_content.side_effect = Exception("model not found")

        with self.assertRaisesRegex(LLMError, "GEMINI_MODEL"):
            GeminiGenerator().generate("question", ["context"])


class GeminiConfigurationTests(unittest.TestCase):
    def test_settings_load_gemini_fields(self) -> None:
        settings = Settings(
            youtube_api_key="youtube-key",
            gemini_api_key="gemini-key",
            gemini_model="gemini-test",
            llm_provider="gemini",
        )
        self.assertEqual(settings.gemini_api_key, "gemini-key")
        self.assertEqual(settings.gemini_model, "gemini-test")

    @patch("src.services.chat.ChromaVectorStore")
    @patch("src.services.chat.SimpleRetriever")
    @patch("src.services.chat.LocalEmbedder")
    @patch("src.services.chat.GeminiGenerator")
    def test_chat_service_selects_gemini(
        self, generator_class: Mock, _embedder: Mock, _retriever: Mock, _store: Mock
    ) -> None:
        settings = SimpleNamespace(
            should_local_embed=True,
            embedding_model="local-model",
            chroma_db_path="data/test",
            llm_provider="gemini",
            gemini_api_key="gemini-key",
            gemini_model="gemini-test",
        )
        service = ChatService(settings=settings)

        self.assertIs(service.generator, generator_class.return_value)
        generator_class.assert_called_once_with(model="gemini-test", api_key="gemini-key")

    def test_chat_service_missing_gemini_key_is_clear(self) -> None:
        service = ChatService.__new__(ChatService)
        service.settings = SimpleNamespace(
            should_local_embed=True,
            embedding_model="local-model",
            chroma_db_path="data/test",
            llm_provider="gemini",
            gemini_api_key=None,
        )
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            # Exercise the real constructor branch without initializing unrelated clients.
            with patch("src.services.chat.BaseService.__init__", lambda instance, settings: setattr(instance, "settings", settings)), \
                 patch("src.services.chat.LocalEmbedder"), patch("src.services.chat.ChromaVectorStore"), \
                 patch("src.services.chat.SimpleRetriever"):
                ChatService(settings=service.settings)


if __name__ == "__main__":
    unittest.main()
