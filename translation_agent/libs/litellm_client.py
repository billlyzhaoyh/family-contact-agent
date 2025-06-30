import os
import logging
from typing import Dict, Any, Optional, List
from litellm import completion, acompletion

from translation_agent.libs.litellm_model import LiteLLMModel, ModelConfig
from translation_agent.libs.config import LiteLLMConfig


class LiteLLMClient:
    """
    LiteLLM client for making LLM API calls with support for multiple providers
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize LiteLLM client
        
        Args:
            api_key: API key for the provider (can also be set via environment variables)
            base_url: Base URL for the API (for custom endpoints)
            provider: Specific provider to use (auto-detected if not specified)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider or LiteLLMConfig.get_default_provider()
        self.logger = logging.getLogger(__name__)
        
        # Set up environment variables for LiteLLM
        self._setup_environment()
        
        # Validate configuration
        if not LiteLLMConfig.validate_provider_config(self.provider):
            self.logger.warning(f"Provider {self.provider} configuration not found. Available providers: {LiteLLMConfig.list_available_providers()}")
    
    def _setup_environment(self):
        """
        Set up environment variables for different providers
        """
        # Set default API key if provided
        if self.api_key:
            if self.provider == "openai":
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif self.provider == "cohere":
                os.environ["COHERE_API_KEY"] = self.api_key
            elif self.provider == "google":
                os.environ["GOOGLE_API_KEY"] = self.api_key
            elif self.provider == "mistral":
                os.environ["MISTRAL_API_KEY"] = self.api_key
        
        # Set base URL if provided (for custom endpoints)
        if self.base_url:
            if self.provider == "openai":
                os.environ["OPENAI_API_BASE"] = self.base_url
            elif self.provider == "anthropic":
                os.environ["ANTHROPIC_API_BASE"] = self.base_url
    
    def invoke_model(
        self,
        prompt: str,
        system_msg: str = "You are a helpful assistant.",
        model: LiteLLMModel = LiteLLMModel.DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Invoke a model with the given prompt and system message
        
        Args:
            prompt: The user prompt
            system_msg: The system message
            model: The model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The generated response text
        """
        try:
            # Prepare messages in OpenAI format
            messages = self._prepare_messages(prompt, system_msg)
            
            # Get model configuration
            provider = ModelConfig.get_provider_from_model(model)
            config = ModelConfig.get_config(model, provider)
            
            # Override with provided parameters
            if temperature is not None:
                config["temperature"] = temperature
            if max_tokens is not None:
                config["max_tokens"] = max_tokens
            
            # Add any additional kwargs
            config.update(kwargs)
            
            # Make the API call
            response = completion(
                model=model.value,
                messages=messages,
                **config
            )
            
            # Extract the response text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("No response content received")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    async def invoke_model_async(
        self,
        prompt: str,
        system_msg: str = "You are a helpful assistant.",
        model: LiteLLMModel = LiteLLMModel.DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async version of invoke_model
        """
        try:
            # Prepare messages in OpenAI format
            messages = self._prepare_messages(prompt, system_msg)
            
            # Get model configuration
            provider = ModelConfig.get_provider_from_model(model)
            config = ModelConfig.get_config(model, provider)
            
            # Override with provided parameters
            if temperature is not None:
                config["temperature"] = temperature
            if max_tokens is not None:
                config["max_tokens"] = max_tokens
            
            # Add any additional kwargs
            config.update(kwargs)
            
            # Make the async API call
            response = await acompletion(
                model=model.value,
                messages=messages,
                **config
            )
            
            # Extract the response text
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("No response content received")
                
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    def _prepare_messages(self, prompt: str, system_msg: str) -> List[Dict[str, str]]:
        """
        Prepare messages in OpenAI-compatible format
        
        Args:
            prompt: The user prompt
            system_msg: The system message
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if provided
        if system_msg:
            messages.append({
                "role": "system",
                "content": system_msg
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def get_model_info(self, model: LiteLLMModel) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model: The model to get info for
            
        Returns:
            Dictionary with model information
        """
        provider = ModelConfig.get_provider_from_model(model)
        config = ModelConfig.get_config(model, provider)
        
        return {
            "model": model.value,
            "provider": provider,
            "config": config
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their configurations
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for model in LiteLLMModel:
            models.append(self.get_model_info(model))
        return models
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider configuration
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.provider,
            "config": LiteLLMConfig.get_provider_config(self.provider),
            "is_valid": LiteLLMConfig.validate_provider_config(self.provider)
        } 