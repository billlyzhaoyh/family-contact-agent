from enum import Enum
from typing import Dict, Any


class LiteLLMModel(Enum):
    """
    LiteLLM Model configurations supporting multiple providers
    See: https://docs.litellm.ai/docs/providers for supported providers
    """

    # Default model - can be configured via environment variables
    DEFAULT_MODEL = "gpt-4o"

    # Anthropic Claude models (via Anthropic API)
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Azure OpenAI models (when using Azure)
    AZURE_GPT_4 = "azure/gpt-4"
    AZURE_GPT_35_TURBO = "azure/gpt-35-turbo"

    # Bedrock models (when using AWS Bedrock)
    BEDROCK_CLAUDE_3_5_SONNET = "bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0"
    BEDROCK_CLAUDE_3_SONNET = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    BEDROCK_CLAUDE_3_HAIKU = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

    # Cohere models
    COHERE_COMMAND = "command"
    COHERE_COMMAND_LIGHT = "command-light"

    # Google models (Vertex AI)
    GEMINI_PRO = "gemini/gemini-pro"
    GEMINI_FLASH = "gemini/gemini-1.5-flash"

    # Mistral models
    MISTRAL_LARGE = "mistral/mistral-large-latest"
    MISTRAL_MEDIUM = "mistral/mistral-medium-latest"
    MISTRAL_SMALL = "mistral/mistral-small-latest"


class ModelConfig:
    """
    Configuration for different model providers and their settings
    """

    # Default configuration for translation tasks
    DEFAULT_CONFIG = {
        "temperature": 0.3,
        "max_tokens": 4000,
        "top_p": 1.0,
    }

    # Provider-specific configurations
    PROVIDER_CONFIGS = {
        "anthropic": {
            "temperature": 0.3,
            "max_tokens": 4000,
            "top_p": 1.0,
        },
        "openai": {
            "temperature": 0.3,
            "max_tokens": 4000,
            "top_p": 1.0,
        },
        "bedrock": {
            "temperature": 0.3,
            "max_tokens": 4000,
            "top_p": 1.0,
        },
        "cohere": {
            "temperature": 0.3,
            "max_tokens": 4000,
            "top_p": 1.0,
        },
    }

    @classmethod
    def get_config(cls, model: LiteLLMModel, provider: str = None) -> Dict[str, Any]:
        """
        Get configuration for a specific model and provider
        """
        config = cls.DEFAULT_CONFIG.copy()

        if provider and provider in cls.PROVIDER_CONFIGS:
            config.update(cls.PROVIDER_CONFIGS[provider])

        return config

    @classmethod
    def get_provider_from_model(cls, model: LiteLLMModel) -> str:
        """
        Extract provider from model identifier
        """
        model_str = model.value

        if model_str.startswith("bedrock/"):
            return "bedrock"
        elif model_str.startswith("azure/"):
            return "azure"
        elif model_str.startswith("gemini/"):
            return "google"
        elif model_str.startswith("mistral/"):
            return "mistral"
        elif model_str.startswith("claude-"):
            return "anthropic"
        elif model_str.startswith("gpt-"):
            return "openai"
        elif model_str.startswith("command"):
            return "cohere"
        else:
            return "openai"  # default fallback
