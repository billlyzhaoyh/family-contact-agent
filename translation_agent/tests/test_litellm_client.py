"""
Unit tests for translation_agent.libs.litellm_client module.
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, patch

from translation_agent.libs.litellm_client import LiteLLMClient
from translation_agent.libs.litellm_model import LiteLLMModel


@pytest.mark.asyncio
class TestLiteLLMClient:
    """Test cases for LiteLLMClient class."""

    async def test_init_default_parameters(self, mock_environment):
        """Test client initialization with default parameters."""
        client = LiteLLMClient()

        assert client.api_key is None
        assert client.base_url is None
        assert client.provider == "openai"  # Default provider
        assert client.logger is not None

    async def test_init_with_parameters(self):
        """Test client initialization with custom parameters."""
        with patch.dict(os.environ, {}, clear=True):
            client = LiteLLMClient(
                api_key="test_key",
                base_url="https://api.test.com",
                provider="anthropic",
            )

            assert client.api_key == "test_key"
            assert client.base_url == "https://api.test.com"
            assert client.provider == "anthropic"

    async def test_setup_environment_openai(self):
        """Test environment setup for OpenAI provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(
                api_key="test_openai_key",
                base_url="https://api.openai.com",
                provider="openai",
            )

            assert os.environ["OPENAI_API_KEY"] == "test_openai_key"
            assert os.environ["OPENAI_API_BASE"] == "https://api.openai.com"

    async def test_setup_environment_anthropic(self):
        """Test environment setup for Anthropic provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(
                api_key="test_anthropic_key",
                base_url="https://api.anthropic.com",
                provider="anthropic",
            )

            assert os.environ["ANTHROPIC_API_KEY"] == "test_anthropic_key"
            assert os.environ["ANTHROPIC_API_BASE"] == "https://api.anthropic.com"

    async def test_setup_environment_cohere(self):
        """Test environment setup for Cohere provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(api_key="test_cohere_key", provider="cohere")

            assert os.environ["COHERE_API_KEY"] == "test_cohere_key"

    async def test_setup_environment_google(self):
        """Test environment setup for Google provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(api_key="test_google_key", provider="google")

            assert os.environ["GOOGLE_API_KEY"] == "test_google_key"

    async def test_setup_environment_mistral(self):
        """Test environment setup for Mistral provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(api_key="test_mistral_key", provider="mistral")

            assert os.environ["MISTRAL_API_KEY"] == "test_mistral_key"

    async def test_setup_environment_unknown_provider(self):
        """Test environment setup for unknown provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(provider="unknown_provider")

            # Should not set any environment variables for unknown provider
            assert "UNKNOWN_PROVIDER_API_KEY" not in os.environ

    @patch("translation_agent.libs.litellm_client.completion")
    async def test_invoke_model_success(self, mock_completion, mock_litellm_response):
        """Test successful model invocation."""
        mock_completion.return_value = mock_litellm_response

        client = LiteLLMClient()
        result = client.invoke_model(
            prompt="Test prompt",
            system_msg="Test system message",
            model=LiteLLMModel.GPT_4O,
        )

        assert result == "Mocked translation response"
        mock_completion.assert_called_once()

    @patch("translation_agent.libs.litellm_client.completion")
    async def test_invoke_model_with_custom_parameters(
        self, mock_completion, mock_litellm_response
    ):
        """Test model invocation with custom parameters."""
        mock_completion.return_value = mock_litellm_response

        client = LiteLLMClient()
        result = client.invoke_model(
            prompt="Test prompt",
            system_msg="Test system message",
            model=LiteLLMModel.GPT_4O,
            temperature=0.7,
            max_tokens=1000,
            custom_param="test_value",
        )

        assert result == "Mocked translation response"

        # Verify the call arguments
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 1000
        assert call_args[1]["custom_param"] == "test_value"

    @patch("translation_agent.libs.litellm_client.completion")
    async def test_invoke_model_empty_response(self, mock_completion):
        """Test model invocation with empty response."""
        mock_response = Mock()
        mock_response.choices = []
        mock_completion.return_value = mock_response

        client = LiteLLMClient()

        with pytest.raises(ValueError, match="No response content received"):
            client.invoke_model(prompt="Test prompt", model=LiteLLMModel.GPT_4O)

    @patch("translation_agent.libs.litellm_client.completion")
    async def test_invoke_model_api_error(self, mock_completion):
        """Test model invocation with API error."""
        mock_completion.side_effect = Exception("API Error")

        client = LiteLLMClient()

        with pytest.raises(Exception, match="API Error"):
            client.invoke_model(prompt="Test prompt", model=LiteLLMModel.GPT_4O)

    @patch("translation_agent.libs.litellm_client.acompletion")
    async def test_invoke_model_async_success(
        self, mock_acompletion, mock_litellm_response
    ):
        """Test successful async model invocation."""
        mock_acompletion.return_value = mock_litellm_response

        client = LiteLLMClient()
        result = await client.invoke_model_async(
            prompt="Test prompt",
            system_msg="Test system message",
            model=LiteLLMModel.GPT_4O,
        )

        assert result == "Mocked translation response"
        mock_acompletion.assert_called_once()

    @patch("translation_agent.libs.litellm_client.acompletion")
    async def test_invoke_model_async_error(self, mock_acompletion):
        """Test async model invocation with error."""
        mock_acompletion.side_effect = Exception("Async API Error")

        client = LiteLLMClient()

        with pytest.raises(Exception, match="Async API Error"):
            await client.invoke_model_async(
                prompt="Test prompt", model=LiteLLMModel.GPT_4O
            )

    async def test_prepare_messages_with_system(self):
        """Test message preparation with system message."""
        client = LiteLLMClient()
        messages = client._prepare_messages("User prompt", "System message")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    async def test_prepare_messages_without_system(self):
        """Test message preparation without system message."""
        client = LiteLLMClient()
        messages = client._prepare_messages("User prompt", "")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User prompt"

    async def test_prepare_messages_none_system(self):
        """Test message preparation with None system message."""
        client = LiteLLMClient()
        messages = client._prepare_messages("User prompt", None)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User prompt"

    async def test_get_model_info(self):
        """Test getting model information."""
        client = LiteLLMClient()
        info = client.get_model_info(LiteLLMModel.GPT_4O)

        assert info["model"] == "gpt-4o"
        assert info["provider"] == "openai"
        assert "config" in info
        assert info["config"]["temperature"] == 0.3

    async def test_get_model_info_anthropic(self):
        """Test getting model information for Anthropic model."""
        client = LiteLLMClient()
        info = client.get_model_info(LiteLLMModel.CLAUDE_3_SONNET)

        assert info["model"] == "claude-3-sonnet-20240229"
        assert info["provider"] == "anthropic"
        assert "config" in info

    async def test_list_available_models(self):
        """Test listing available models."""
        client = LiteLLMClient()
        models = client.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check structure of model info
        for model_info in models:
            assert "model" in model_info
            assert "provider" in model_info
            assert "config" in model_info

    async def test_get_provider_info(self):
        """Test getting provider information."""
        client = LiteLLMClient()
        info = client.get_provider_info()

        assert "provider" in info
        assert "config" in info
        assert "is_valid" in info

    async def test_client_logger_initialization(self):
        """Test that logger is properly initialized."""
        client = LiteLLMClient()
        assert client.logger is not None
        assert client.logger.name == "translation_agent.libs.litellm_client"

    @patch("translation_agent.libs.litellm_client.LiteLLMConfig")
    async def test_client_validation_warning(self, mock_config):
        """Test client initialization with invalid provider configuration."""
        mock_config.validate_provider_config.return_value = False
        mock_config.list_available_providers.return_value = ["openai"]

        with patch("translation_agent.libs.litellm_client.logging") as mock_logging:
            mock_logger = Mock()
            mock_logging.getLogger.return_value = mock_logger

            LiteLLMClient(provider="invalid_provider")

            mock_logger.warning.assert_called_once()

    async def test_client_with_custom_base_url_only(self):
        """Test client initialization with only base_url."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(base_url="https://custom.api.com", provider="openai")

            assert os.environ["OPENAI_API_BASE"] == "https://custom.api.com"

    async def test_client_with_api_key_only(self):
        """Test client initialization with only api_key."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMClient(api_key="test_key_only", provider="openai")

            assert os.environ["OPENAI_API_KEY"] == "test_key_only"
            assert "OPENAI_API_BASE" not in os.environ

    # Additional async-specific tests
    @patch("translation_agent.libs.litellm_client.acompletion")
    async def test_invoke_model_async_with_async_mocks(
        self, mock_acompletion, mock_litellm_response
    ):
        """Test async model invocation with AsyncMock for better async testing."""
        mock_acompletion.return_value = mock_litellm_response

        client = LiteLLMClient()
        result = await client.invoke_model_async(
            prompt="Async test prompt",
            system_msg="Async system message",
            model=LiteLLMModel.GPT_4O,
            temperature=0.5,
        )

        assert result == "Mocked translation response"
        mock_acompletion.assert_called_once()

    async def test_concurrent_model_invocations(self):
        """Test that multiple model invocations can be executed concurrently."""
        with patch(
            "translation_agent.libs.litellm_client.acompletion"
        ) as mock_acompletion:
            # Create mock responses for different calls
            mock_response1 = Mock()
            mock_response1.choices = [Mock()]
            mock_response1.choices[0].message.content = "Response 1"

            mock_response2 = Mock()
            mock_response2.choices = [Mock()]
            mock_response2.choices[0].message.content = "Response 2"

            mock_response3 = Mock()
            mock_response3.choices = [Mock()]
            mock_response3.choices[0].message.content = "Response 3"

            # Configure mock to return different responses
            mock_acompletion.side_effect = [
                mock_response1,
                mock_response2,
                mock_response3,
            ]

            client = LiteLLMClient()

            # Create concurrent tasks
            tasks = [
                client.invoke_model_async(
                    prompt=f"Prompt {i}", model=LiteLLMModel.GPT_4O
                )
                for i in range(1, 4)
            ]

            results = await asyncio.gather(*tasks)

            assert results == ["Response 1", "Response 2", "Response 3"]
            assert mock_acompletion.call_count == 3

    @patch("translation_agent.libs.litellm_client.acompletion")
    async def test_invoke_model_async_with_custom_parameters(
        self, mock_acompletion, mock_litellm_response
    ):
        """Test async model invocation with custom parameters."""
        mock_acompletion.return_value = mock_litellm_response

        client = LiteLLMClient()
        result = await client.invoke_model_async(
            prompt="Test prompt",
            system_msg="Test system message",
            model=LiteLLMModel.GPT_4O,
            temperature=0.8,
            max_tokens=500,
            custom_async_param="async_value",
        )

        assert result == "Mocked translation response"

        # Verify the call arguments
        call_args = mock_acompletion.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["custom_async_param"] == "async_value"

    async def test_mixed_sync_async_operations(self):
        """Test mixing sync and async operations in the same test."""
        with patch(
            "translation_agent.libs.litellm_client.completion"
        ) as mock_completion:
            with patch(
                "translation_agent.libs.litellm_client.acompletion"
            ) as mock_acompletion:
                # Setup mock responses
                sync_response = Mock()
                sync_response.choices = [Mock()]
                sync_response.choices[0].message.content = "Sync response"

                async_response = Mock()
                async_response.choices = [Mock()]
                async_response.choices[0].message.content = "Async response"

                mock_completion.return_value = sync_response
                mock_acompletion.return_value = async_response

                client = LiteLLMClient()

                # Test sync operation
                sync_result = client.invoke_model(
                    prompt="Sync prompt", model=LiteLLMModel.GPT_4O
                )

                # Test async operation
                async_result = await client.invoke_model_async(
                    prompt="Async prompt", model=LiteLLMModel.GPT_4O
                )

                assert sync_result == "Sync response"
                assert async_result == "Async response"
                mock_completion.assert_called_once()
                mock_acompletion.assert_called_once()

    @patch("translation_agent.libs.litellm_client.acompletion")
    async def test_invoke_model_async_empty_response(self, mock_acompletion):
        """Test async model invocation with empty response."""
        mock_response = Mock()
        mock_response.choices = []
        mock_acompletion.return_value = mock_response

        client = LiteLLMClient()

        with pytest.raises(ValueError, match="No response content received"):
            await client.invoke_model_async(
                prompt="Test prompt", model=LiteLLMModel.GPT_4O
            )

    async def test_async_context_manager_support(self):
        """Test that client can be used in async context managers."""
        client = LiteLLMClient()

        # Test that client can be used in async context
        async with asyncio.timeout(1.0):  # Simple async context
            result = client.get_model_info(LiteLLMModel.GPT_4O)
            assert result["model"] == "gpt-4o"
