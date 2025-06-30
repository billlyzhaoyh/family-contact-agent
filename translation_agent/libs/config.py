import os
from typing import Optional, Dict, Any


class LiteLLMConfig:
    """
    Configuration management for LiteLLM providers
    """
    
    # Environment variable mappings for different providers
    PROVIDER_ENV_VARS = {
        "openai": {
            "api_key": "OPENAI_API_KEY",
            "base_url": "OPENAI_API_BASE",
            "organization": "OPENAI_ORGANIZATION",
        },
        "anthropic": {
            "api_key": "ANTHROPIC_API_KEY",
            "base_url": "ANTHROPIC_API_BASE",
        },
        "azure": {
            "api_key": "AZURE_API_KEY",
            "base_url": "AZURE_API_BASE",
            "api_version": "AZURE_API_VERSION",
        },
        "bedrock": {
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "aws_region": "AWS_DEFAULT_REGION",
        },
        "cohere": {
            "api_key": "COHERE_API_KEY",
        },
        "google": {
            "api_key": "GOOGLE_API_KEY",
            "project_id": "GOOGLE_PROJECT_ID",
        },
        "mistral": {
            "api_key": "MISTRAL_API_KEY",
        },
    }
    
    @classmethod
    def setup_environment(cls, provider: str, config: Dict[str, Any]):
        """
        Set up environment variables for a specific provider
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            config: Configuration dictionary with API keys and settings
        """
        if provider not in cls.PROVIDER_ENV_VARS:
            raise ValueError(f"Unsupported provider: {provider}")
        
        env_vars = cls.PROVIDER_ENV_VARS[provider]
        
        for key, env_var in env_vars.items():
            if key in config:
                os.environ[env_var] = str(config[key])
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Dict[str, str]:
        """
        Get current environment configuration for a provider
        
        Args:
            provider: The provider name
            
        Returns:
            Dictionary with current environment variable values
        """
        if provider not in cls.PROVIDER_ENV_VARS:
            raise ValueError(f"Unsupported provider: {provider}")
        
        env_vars = cls.PROVIDER_ENV_VARS[provider]
        config = {}
        
        for key, env_var in env_vars.items():
            value = os.environ.get(env_var)
            if value:
                config[key] = value
        
        return config
    
    @classmethod
    def validate_provider_config(cls, provider: str) -> bool:
        """
        Validate that required environment variables are set for a provider
        
        Args:
            provider: The provider name
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if provider not in cls.PROVIDER_ENV_VARS:
            return False
        
        env_vars = cls.PROVIDER_ENV_VARS[provider]
        required_vars = [env_vars.get("api_key") or env_vars.get("aws_access_key_id")]
        
        for env_var in required_vars:
            if not os.environ.get(env_var):
                return False
        
        return True
    
    @classmethod
    def get_default_provider(cls) -> str:
        """
        Get the default provider based on available environment variables
        
        Returns:
            Provider name or 'openai' as fallback
        """
        # Check for common providers in order of preference
        providers = ["openai", "anthropic", "bedrock", "azure", "cohere", "google", "mistral"]
        
        for provider in providers:
            if cls.validate_provider_config(provider):
                return provider
        
        return "openai"  # Default fallback
    
    @classmethod
    def list_available_providers(cls) -> list:
        """
        List all providers that have valid configuration
        
        Returns:
            List of provider names with valid configuration
        """
        available = []
        
        for provider in cls.PROVIDER_ENV_VARS.keys():
            if cls.validate_provider_config(provider):
                available.append(provider)
        
        return available 