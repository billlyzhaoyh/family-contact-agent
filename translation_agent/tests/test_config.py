"""
Unit tests for translation_agent.libs.config module.
"""

import os
import pytest
from unittest.mock import patch

from translation_agent.libs.config import LiteLLMConfig


class TestLiteLLMConfig:
    """Test cases for LiteLLMConfig class."""

    def test_provider_env_vars_structure(self):
        """Test that PROVIDER_ENV_VARS contains expected providers."""
        expected_providers = {
            "openai",
            "anthropic",
            "azure",
            "bedrock",
            "cohere",
            "google",
            "mistral",
        }
        assert set(LiteLLMConfig.PROVIDER_ENV_VARS.keys()) == expected_providers

    def test_setup_environment_valid_provider(self, sample_config):
        """Test setting up environment variables for a valid provider."""
        with patch.dict(os.environ, {}, clear=True):
            LiteLLMConfig.setup_environment("openai", sample_config)

            assert os.environ["OPENAI_API_KEY"] == "test_api_key_12345"
            assert os.environ["OPENAI_API_BASE"] == "https://api.test.com/v1"
            assert os.environ["OPENAI_ORGANIZATION"] == "test_org"

    def test_setup_environment_invalid_provider(self, sample_config):
        """Test that setup_environment raises ValueError for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid_provider"):
            LiteLLMConfig.setup_environment("invalid_provider", sample_config)

    def test_setup_environment_partial_config(self):
        """Test setting up environment with partial configuration."""
        partial_config = {"api_key": "test_key"}

        with patch.dict(os.environ, {}, clear=True):
            LiteLLMConfig.setup_environment("anthropic", partial_config)

            assert os.environ["ANTHROPIC_API_KEY"] == "test_key"
            # Should not set base_url if not provided
            assert "ANTHROPIC_API_BASE" not in os.environ

    def test_get_provider_config_valid_provider(self, mock_environment):
        """Test getting provider configuration for valid provider."""
        config = LiteLLMConfig.get_provider_config("openai")

        assert config["api_key"] == "test_openai_key"
        assert "base_url" not in config  # Not set in mock environment

    def test_get_provider_config_invalid_provider(self):
        """Test that get_provider_config raises ValueError for invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid_provider"):
            LiteLLMConfig.get_provider_config("invalid_provider")

    def test_get_provider_config_empty_environment(self):
        """Test getting provider config when environment variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = LiteLLMConfig.get_provider_config("openai")
            assert config == {}

    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openai", True),
            ("anthropic", True),
            ("azure", True),
            ("invalid_provider", False),
        ],
    )
    def test_validate_provider_config(self, mock_environment, provider, expected):
        """Test provider configuration validation."""
        result = LiteLLMConfig.validate_provider_config(provider)
        assert result == expected

    def test_validate_provider_config_missing_api_key(self):
        """Test validation when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = LiteLLMConfig.validate_provider_config("openai")
            assert result is False

    def test_get_default_provider_with_available_providers(self, mock_environment):
        """Test getting default provider when multiple providers are available."""
        # With mock environment, openai should be the first available
        provider = LiteLLMConfig.get_default_provider()
        assert provider == "openai"

    def test_get_default_provider_no_available_providers(self):
        """Test getting default provider when no providers are configured."""
        with patch.dict(os.environ, {}, clear=True):
            provider = LiteLLMConfig.get_default_provider()
            assert provider == "openai"  # Default fallback

    def test_list_available_providers(self, mock_environment):
        """Test listing available providers."""
        available = LiteLLMConfig.list_available_providers()

        # Should include providers with valid config
        assert "openai" in available
        assert "anthropic" in available
        assert "azure" in available

        # Should not include invalid providers
        assert "invalid_provider" not in available

    def test_list_available_providers_empty_environment(self):
        """Test listing available providers when no providers are configured."""
        with patch.dict(os.environ, {}, clear=True):
            available = LiteLLMConfig.list_available_providers()
            assert available == []

    def test_bedrock_provider_validation(self):
        """Test Bedrock provider validation with AWS credentials."""
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "test_access_key",
                "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        ):
            result = LiteLLMConfig.validate_provider_config("bedrock")
            assert result is True

    def test_bedrock_provider_validation_missing_credentials(self):
        """Test Bedrock provider validation with missing AWS credentials."""
        with patch.dict(os.environ, {}, clear=True):
            result = LiteLLMConfig.validate_provider_config("bedrock")
            assert result is False

    def test_google_provider_validation(self):
        """Test Google provider validation."""
        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test_google_key", "GOOGLE_PROJECT_ID": "test_project"},
        ):
            result = LiteLLMConfig.validate_provider_config("google")
            assert result is True

    def test_cohere_provider_validation(self):
        """Test Cohere provider validation."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test_cohere_key"}):
            result = LiteLLMConfig.validate_provider_config("cohere")
            assert result is True

    def test_mistral_provider_validation(self):
        """Test Mistral provider validation."""
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_mistral_key"}):
            result = LiteLLMConfig.validate_provider_config("mistral")
            assert result is True
