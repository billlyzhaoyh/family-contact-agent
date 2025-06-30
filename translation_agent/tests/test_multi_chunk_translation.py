"""
Unit tests for translation_agent.libs.multi_chunk_translation module.
"""

from unittest.mock import Mock, patch

from translation_agent.libs.multi_chunk_translation import MultiChunkTranslation
from translation_agent.libs.litellm_model import LiteLLMModel


class TestMultiChunkTranslation:
    """Test cases for MultiChunkTranslation class."""

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_init_default_parameters(self, mock_split_text, mock_client_class):
        """Test MultiChunkTranslation initialization with default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1", "chunk2"]

        translation = MultiChunkTranslation(
            source_text="Long text",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        assert translation._MultiChunkTranslation__source_text_chunks == [
            "chunk1",
            "chunk2",
        ]
        assert translation._MultiChunkTranslation__source_lang == "English"
        assert translation._MultiChunkTranslation__target_lang == ("Spanish",)
        assert translation._MultiChunkTranslation__country == "Spain"
        assert translation._MultiChunkTranslation__litellm_client == mock_client

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_init_without_country(self, mock_split_text, mock_client_class):
        """Test MultiChunkTranslation initialization without country."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]

        translation = MultiChunkTranslation(
            source_text="Text", source_lang="English", target_lang="Spanish"
        )

        assert translation._MultiChunkTranslation__country == ""

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_set_init_model(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        result = translation.set_init_model(LiteLLMModel.GPT_4O)
        assert result is translation
        assert translation._init_translation_model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_set_reflect_on_model(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        result = translation.set_reflect_on_model(LiteLLMModel.GPT_4O)
        assert result is translation
        assert translation._reflect_on_translation_model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_set_improve_model(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        result = translation.set_improve_model(LiteLLMModel.GPT_4O)
        assert result is translation
        assert translation._improve_translation_model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_get_init_translation_model_default(
        self, mock_split_text, mock_client_class
    ):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        model = translation._get_init_translation_model()
        assert model == LiteLLMModel.CLAUDE_3_SONNET

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_get_reflect_on_translation_model_default(
        self, mock_split_text, mock_client_class
    ):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        model = translation._get_reflect_on_translation_model()
        assert model == LiteLLMModel.CLAUDE_3_SONNET

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_get_improve_translation_model_default(
        self, mock_split_text, mock_client_class
    ):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1"]
        translation = MultiChunkTranslation("Text", "English", "Spanish")
        model = translation._get_improve_translation_model()
        assert model == LiteLLMModel.CLAUDE_3_SONNET

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_do_complete_flow(self, mock_split_text, mock_client_class):
        """Test complete translation flow."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "init1",
            "init2",  # __init_translation (2 chunks)
            "reflect1",
            "reflect2",  # __reflect_on_translation (2 chunks)
            "improve1",
            "improve2",  # __improve_translation (2 chunks)
        ]
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1", "chunk2"]
        translation = MultiChunkTranslation("Long text", "English", "Spanish", "Spain")
        result = translation.do()
        assert mock_client.invoke_model.call_count == 6
        assert result == "improve1improve2"

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_do_with_custom_models(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "init1",
            "init2",
            "reflect1",
            "reflect2",
            "improve1",
            "improve2",
        ]
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1", "chunk2"]
        translation = MultiChunkTranslation("Long text", "English", "Spanish")
        translation.set_init_model(LiteLLMModel.GPT_4O)
        translation.set_reflect_on_model(LiteLLMModel.GPT_4O)
        translation.set_improve_model(LiteLLMModel.GPT_4O)
        result = translation.do()
        assert mock_client.invoke_model.call_count == 6
        assert result == "improve1improve2"

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_do_preserves_original_attributes(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "init1",
            "init2",
            "reflect1",
            "reflect2",
            "improve1",
            "improve2",
        ]
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["chunk1", "chunk2"]
        translation = MultiChunkTranslation("Long text", "English", "Spanish", "Spain")
        original_chunks = translation._MultiChunkTranslation__source_text_chunks.copy()
        original_lang = translation._MultiChunkTranslation__source_lang
        original_target = translation._MultiChunkTranslation__target_lang
        original_country = translation._MultiChunkTranslation__country
        result = translation.do()
        assert translation._MultiChunkTranslation__source_text_chunks == original_chunks
        assert translation._MultiChunkTranslation__source_lang == original_lang
        assert translation._MultiChunkTranslation__target_lang == original_target
        assert translation._MultiChunkTranslation__country == original_country
        assert result == "improve1improve2"

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_do_with_empty_chunks(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ["", "", "", "", "", ""]
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["", ""]
        translation = MultiChunkTranslation("", "English", "Spanish")
        result = translation.do()
        assert mock_client.invoke_model.call_count == 6
        assert result == ""

    @patch("translation_agent.libs.multi_chunk_translation.LiteLLMClient")
    @patch("translation_agent.libs.multi_chunk_translation.split_text_by_token_size")
    def test_do_with_special_characters(self, mock_split_text, mock_client_class):
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "¡Hola!",
            "¿Qué tal?",
            "Good",
            "Great",
            "¡Hola!",
            "¿Qué tal?",
        ]
        mock_client_class.return_value = mock_client
        mock_split_text.return_value = ["Hello!", "How are you?"]
        translation = MultiChunkTranslation("Hello! How are you?", "English", "Spanish")
        result = translation.do()
        assert mock_client.invoke_model.call_count == 6
        assert result == "¡Hola!¿Qué tal?"
