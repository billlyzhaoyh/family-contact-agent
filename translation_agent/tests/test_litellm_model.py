"""
Unit tests for translation_agent.libs.litellm_model module.
"""

import pytest

from translation_agent.libs.litellm_model import LiteLLMModel, ModelConfig


class TestLiteLLMModel:
    """Test cases for LiteLLMModel enum."""

    def test_default_model(self):
        """Test that DEFAULT_MODEL has the expected value."""
        assert LiteLLMModel.DEFAULT_MODEL.value == "gpt-4o"

    def test_anthropic_models(self):
        """Test Anthropic Claude model values."""
        assert LiteLLMModel.CLAUDE_3_5_SONNET.value == "claude-3-5-sonnet-20241022"
        assert LiteLLMModel.CLAUDE_3_5_HAIKU.value == "claude-3-5-haiku-20241022"
        assert LiteLLMModel.CLAUDE_3_OPUS.value == "claude-3-opus-20240229"
        assert LiteLLMModel.CLAUDE_3_SONNET.value == "claude-3-sonnet-20240229"
        assert LiteLLMModel.CLAUDE_3_HAIKU.value == "claude-3-haiku-20240307"

    def test_openai_models(self):
        """Test OpenAI model values."""
        assert LiteLLMModel.GPT_4O.value == "gpt-4o"
        assert LiteLLMModel.GPT_4O_MINI.value == "gpt-4o-mini"
        assert LiteLLMModel.GPT_4_TURBO.value == "gpt-4-turbo"
        assert LiteLLMModel.GPT_3_5_TURBO.value == "gpt-3.5-turbo"

    def test_azure_models(self):
        """Test Azure OpenAI model values."""
        assert LiteLLMModel.AZURE_GPT_4.value == "azure/gpt-4"
        assert LiteLLMModel.AZURE_GPT_35_TURBO.value == "azure/gpt-35-turbo"

    def test_bedrock_models(self):
        """Test AWS Bedrock model values."""
        assert (
            LiteLLMModel.BEDROCK_CLAUDE_3_5_SONNET.value
            == "bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0"
        )
        assert (
            LiteLLMModel.BEDROCK_CLAUDE_3_SONNET.value
            == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        )
        assert (
            LiteLLMModel.BEDROCK_CLAUDE_3_HAIKU.value
            == "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
        )

    def test_cohere_models(self):
        """Test Cohere model values."""
        assert LiteLLMModel.COHERE_COMMAND.value == "command"
        assert LiteLLMModel.COHERE_COMMAND_LIGHT.value == "command-light"

    def test_google_models(self):
        """Test Google Vertex AI model values."""
        assert LiteLLMModel.GEMINI_PRO.value == "gemini/gemini-pro"
        assert LiteLLMModel.GEMINI_FLASH.value == "gemini/gemini-1.5-flash"

    def test_mistral_models(self):
        """Test Mistral model values."""
        assert LiteLLMModel.MISTRAL_LARGE.value == "mistral/mistral-large-latest"
        assert LiteLLMModel.MISTRAL_MEDIUM.value == "mistral/mistral-medium-latest"
        assert LiteLLMModel.MISTRAL_SMALL.value == "mistral/mistral-small-latest"


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has the expected structure."""
        default_config = ModelConfig.DEFAULT_CONFIG

        assert "temperature" in default_config
        assert "max_tokens" in default_config
        assert "top_p" in default_config

        assert default_config["temperature"] == 0.3
        assert default_config["max_tokens"] == 4000
        assert default_config["top_p"] == 1.0

    def test_provider_configs_structure(self):
        """Test that PROVIDER_CONFIGS contains expected providers."""
        expected_providers = {"anthropic", "openai", "bedrock", "cohere"}
        assert set(ModelConfig.PROVIDER_CONFIGS.keys()) == expected_providers

    def test_provider_configs_values(self):
        """Test that provider configs have expected values."""
        for provider, config in ModelConfig.PROVIDER_CONFIGS.items():
            assert "temperature" in config
            assert "max_tokens" in config
            assert "top_p" in config

            assert config["temperature"] == 0.3
            assert config["max_tokens"] == 4000
            assert config["top_p"] == 1.0

    @pytest.mark.parametrize(
        "model,expected_provider",
        [
            (LiteLLMModel.BEDROCK_CLAUDE_3_SONNET, "bedrock"),
            (LiteLLMModel.AZURE_GPT_4, "azure"),
            (LiteLLMModel.GEMINI_PRO, "google"),
            (LiteLLMModel.MISTRAL_LARGE, "mistral"),
            (LiteLLMModel.CLAUDE_3_SONNET, "anthropic"),
            (LiteLLMModel.GPT_4O, "openai"),
            (LiteLLMModel.COHERE_COMMAND, "cohere"),
            (LiteLLMModel.DEFAULT_MODEL, "openai"),  # Default fallback
        ],
    )
    def test_get_provider_from_model(self, model, expected_provider):
        """Test provider detection from model identifiers."""
        result = ModelConfig.get_provider_from_model(model)
        assert result == expected_provider

    def test_get_config_default(self):
        """Test getting default configuration without provider."""
        config = ModelConfig.get_config(LiteLLMModel.GPT_4O)

        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 4000
        assert config["top_p"] == 1.0

    def test_get_config_with_provider(self):
        """Test getting configuration with specific provider."""
        config = ModelConfig.get_config(LiteLLMModel.CLAUDE_3_SONNET, "anthropic")

        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 4000
        assert config["top_p"] == 1.0

    def test_get_config_with_invalid_provider(self):
        """Test getting configuration with invalid provider falls back to default."""
        config = ModelConfig.get_config(LiteLLMModel.GPT_4O, "invalid_provider")

        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 4000
        assert config["top_p"] == 1.0

    def test_get_config_returns_copy(self):
        """Test that get_config returns a copy, not the original."""
        config1 = ModelConfig.get_config(LiteLLMModel.GPT_4O)
        config2 = ModelConfig.get_config(LiteLLMModel.GPT_4O)

        # Modify one config
        config1["temperature"] = 0.5

        # The other should remain unchanged
        assert config2["temperature"] == 0.3
        assert config1["temperature"] == 0.5  # Default fallback

    def test_get_config_with_custom_parameters(self):
        """Test that get_config properly merges provider config with default."""
        # Create a mock provider config
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                ModelConfig,
                "PROVIDER_CONFIGS",
                {
                    "test_provider": {
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "custom_param": "test_value",
                    }
                },
            )

            config = ModelConfig.get_config(LiteLLMModel.GPT_4O, "test_provider")

            # Should have provider-specific values
            assert config["temperature"] == 0.7
            assert config["max_tokens"] == 2000
            assert config["custom_param"] == "test_value"

            # Should still have default values for missing keys
            assert config["top_p"] == 1.0

    def test_model_enum_iteration(self):
        """Test that all model enum values are accessible."""
        models = list(LiteLLMModel)
        assert len(models) > 0

        # Test that we can access all model values
        for model in models:
            assert isinstance(model.value, str)
            assert len(model.value) > 0

    def test_model_config_immutability(self):
        """Test that model configs are not accidentally modified."""
        original_default = ModelConfig.DEFAULT_CONFIG.copy()
        original_provider = ModelConfig.PROVIDER_CONFIGS["openai"].copy()

        # Try to modify the configs
        config = ModelConfig.get_config(LiteLLMModel.GPT_4O, "openai")
        config["temperature"] = 0.9

        # Original configs should remain unchanged
        assert ModelConfig.DEFAULT_CONFIG == original_default
        assert ModelConfig.PROVIDER_CONFIGS["openai"] == original_provider
