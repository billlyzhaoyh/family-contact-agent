"""
Unit tests for translation_agent.libs.one_chunk_translation module.
"""

from unittest.mock import Mock, patch

from translation_agent.libs.one_chunk_translation import OneChunkTranslation
from translation_agent.libs.litellm_model import LiteLLMModel


class TestOneChunkTranslation:
    """Test cases for OneChunkTranslation class."""

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_init_default_parameters(self, mock_client_class):
        """Test OneChunkTranslation initialization with default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        assert translation._OneChunkTranslation__source_text == "Hello world"
        assert translation._OneChunkTranslation__source_lang == "English"
        assert translation._OneChunkTranslation__target_lang == "Spanish"
        assert translation._OneChunkTranslation__country == "Spain"
        assert translation._OneChunkTranslation__litellm_client == mock_client

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_init_without_country(self, mock_client_class):
        """Test OneChunkTranslation initialization without country."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        assert translation._OneChunkTranslation__country == ""

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_set_init_model(self, mock_client_class):
        """Test setting init model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation.set_init_model(LiteLLMModel.CLAUDE_3_SONNET)

        assert result is translation  # Should return self for chaining
        assert translation._init_translation_model == LiteLLMModel.CLAUDE_3_SONNET

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_set_reflect_on_model(self, mock_client_class):
        """Test setting reflect on model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation.set_reflect_on_model(LiteLLMModel.GPT_4O)

        assert result is translation
        assert translation._reflect_on_translation_model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_set_improve_model(self, mock_client_class):
        """Test setting improve model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation.set_improve_model(LiteLLMModel.CLAUDE_3_HAIKU)

        assert result is translation
        assert translation._improve_translation_model == LiteLLMModel.CLAUDE_3_HAIKU

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_get_init_translation_model_default(self, mock_client_class):
        """Test getting default init translation model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        model = translation._get_init_translation_model()
        assert model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_get_reflect_on_translation_model_default(self, mock_client_class):
        """Test getting default reflect on translation model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        model = translation._get_reflect_on_translation_model()
        assert model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_get_improve_translation_model_default(self, mock_client_class):
        """Test getting default improve translation model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        model = translation._get_improve_translation_model()
        assert model == LiteLLMModel.GPT_4O

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_init_translation_with_country(self, mock_client_class):
        """Test initial translation step with country."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Initial translation"
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        result = translation._OneChunkTranslation__init_translation()

        # Verify the system message and prompt
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args

        # Check system message
        assert "You are an expert linguist" in call_args[1]["system_msg"]
        assert "English" in call_args[1]["system_msg"]
        assert "Spanish" in call_args[1]["system_msg"]

        # Check prompt
        assert "English to Spanish translation" in call_args[1]["prompt"]
        assert "Hello world" in call_args[1]["prompt"]

        assert result == "Initial translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_init_translation_without_country(self, mock_client_class):
        """Test initial translation step without country."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Initial translation"
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation._OneChunkTranslation__init_translation()

        mock_client.invoke_model.assert_called_once()
        assert result == "Initial translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_reflect_on_translation_with_country(self, mock_client_class):
        """Test reflection step with country."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Reflection suggestions"
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        result = translation._OneChunkTranslation__reflect_on_translation(
            "Initial translation"
        )

        # Verify the system message and prompt
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args

        # Check system message
        assert "You are an expert linguist" in call_args[1]["system_msg"]
        assert "improve the translation" in call_args[1]["system_msg"]

        # Check prompt contains country-specific instructions
        assert "Spain" in call_args[1]["prompt"]
        assert "colloquially spoken in Spain" in call_args[1]["prompt"]

        assert result == "Reflection suggestions"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_reflect_on_translation_without_country(self, mock_client_class):
        """Test reflection step without country."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Reflection suggestions"
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation._OneChunkTranslation__reflect_on_translation(
            "Initial translation"
        )

        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args

        # Check prompt doesn't contain country-specific instructions
        assert "colloquially spoken" not in call_args[1]["prompt"]

        assert result == "Reflection suggestions"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_improve_translation(self, mock_client_class):
        """Test improvement step."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Improved translation"
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        result = translation._OneChunkTranslation__improve_translation(
            pre_translation="Initial translation", reflection="Reflection suggestions"
        )

        # Verify the system message and prompt
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args

        # Check system message
        assert "You are an expert linguist" in call_args[1]["system_msg"]
        assert "translation editing" in call_args[1]["system_msg"]

        # Check prompt contains all required elements
        assert "Initial translation" in call_args[1]["prompt"]
        assert "Reflection suggestions" in call_args[1]["prompt"]
        assert "Hello world" in call_args[1]["prompt"]

        assert result == "Improved translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_do_complete_flow(self, mock_client_class):
        """Test complete translation flow."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "Initial translation",
            "Reflection suggestions",
            "Final improved translation",
        ]
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        result = translation.do()

        # Verify all three steps were called
        assert mock_client.invoke_model.call_count == 3

        # Verify the calls were made in the correct order
        calls = mock_client.invoke_model.call_args_list

        # First call should be init translation
        assert (
            "expert linguist, specializing in translation" in calls[0][1]["system_msg"]
        )

        # Second call should be reflection
        assert "improve the translation" in calls[1][1]["system_msg"]

        # Third call should be improvement
        assert "translation editing" in calls[2][1]["system_msg"]

        assert result == "Final improved translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_do_with_custom_models(self, mock_client_class):
        """Test complete flow with custom models."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "Initial translation",
            "Reflection suggestions",
            "Final improved translation",
        ]
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world", source_lang="English", target_lang="Spanish"
        )

        # Set custom models
        translation.set_init_model(LiteLLMModel.CLAUDE_3_SONNET)
        translation.set_reflect_on_model(LiteLLMModel.GPT_4O)
        translation.set_improve_model(LiteLLMModel.CLAUDE_3_HAIKU)

        result = translation.do()

        # Verify all three steps were called with correct models
        calls = mock_client.invoke_model.call_args_list
        assert calls[0][1]["model"] == LiteLLMModel.CLAUDE_3_SONNET
        assert calls[1][1]["model"] == LiteLLMModel.GPT_4O
        assert calls[2][1]["model"] == LiteLLMModel.CLAUDE_3_HAIKU

        # Check that models were used correctly (this would require checking the model parameter)
        assert mock_client.invoke_model.call_count == 3
        assert result == "Final improved translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_do_preserves_original_attributes(self, mock_client_class):
        """Test that do() doesn't modify original attributes."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "Initial translation",
            "Reflection suggestions",
            "Final improved translation",
        ]
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world",
            source_lang="English",
            target_lang="Spanish",
            country="Spain",
        )

        # Store original values
        original_source_text = translation._OneChunkTranslation__source_text
        original_source_lang = translation._OneChunkTranslation__source_lang
        original_target_lang = translation._OneChunkTranslation__target_lang
        original_country = translation._OneChunkTranslation__country

        result = translation.do()

        # Verify original attributes are preserved
        assert translation._OneChunkTranslation__source_text == original_source_text
        assert translation._OneChunkTranslation__source_lang == original_source_lang
        assert translation._OneChunkTranslation__target_lang == original_target_lang
        assert translation._OneChunkTranslation__country == original_country

        assert result == "Final improved translation"

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_do_with_empty_text(self, mock_client_class):
        """Test translation flow with empty text."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ["", "No suggestions needed", ""]
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="", source_lang="English", target_lang="Spanish"
        )

        result = translation.do()

        assert mock_client.invoke_model.call_count == 3
        assert result == ""

    @patch("translation_agent.libs.one_chunk_translation.LiteLLMClient")
    def test_do_with_special_characters(self, mock_client_class):
        """Test translation flow with special characters."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = [
            "¡Hola mundo!",
            "Good translation",
            "¡Hola mundo!",
        ]
        mock_client_class.return_value = mock_client

        translation = OneChunkTranslation(
            source_text="Hello world!", source_lang="English", target_lang="Spanish"
        )

        result = translation.do()

        assert mock_client.invoke_model.call_count == 3
        assert result == "¡Hola mundo!"
