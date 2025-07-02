"""
Common test fixtures and configuration for translation_agent tests.
"""

import os
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any


@pytest.fixture(autouse=True)
def prevent_real_api_calls():
    """
    Automatically applied to all tests to prevent real API calls.
    This fixture ensures that no actual API requests are made during testing.
    """
    # Clear any real API keys from environment
    with patch.dict(
        os.environ,
        {
            # Clear real API keys
            "OPENAI_API_KEY": "test_key_do_not_use",
            "ANTHROPIC_API_KEY": "test_key_do_not_use",
            "COHERE_API_KEY": "test_key_do_not_use",
            "GOOGLE_API_KEY": "test_key_do_not_use",
            "MISTRAL_API_KEY": "test_key_do_not_use",
            "AZURE_API_KEY": "test_key_do_not_use",
            "AWS_ACCESS_KEY_ID": "test_key_do_not_use",
            "AWS_SECRET_ACCESS_KEY": "test_key_do_not_use",
            # Set test base URLs
            "OPENAI_API_BASE": "https://test.openai.com",
            "ANTHROPIC_API_BASE": "https://test.anthropic.com",
            "AZURE_API_BASE": "https://test.azure.com",
        },
        clear=True,
    ):
        yield


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "api_key": "test_api_key_12345",
        "base_url": "https://api.test.com/v1",
        "organization": "test_org",
    }


@pytest.fixture
def sample_text() -> str:
    """Sample text for translation testing."""
    return "Hello, this is a test message for translation."


@pytest.fixture
def long_text() -> str:
    """Long text for testing chunking functionality."""
    return "This is a very long text that should be split into multiple chunks. " * 100


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM API response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "Mocked translation response"
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_litellm_completion(mock_litellm_response):
    """Mock LiteLLM completion function."""
    with patch("translation_agent.libs.litellm_client.completion") as mock_completion:
        mock_completion.return_value = mock_litellm_response
        yield mock_completion


@pytest.fixture
def mock_litellm_acompletion(mock_litellm_response):
    """Mock LiteLLM async completion function."""
    with patch("translation_agent.libs.litellm_client.acompletion") as mock_acompletion:
        mock_acompletion.return_value = mock_litellm_response
        yield mock_acompletion


@pytest.fixture
def sample_translation_params():
    """Sample parameters for translation testing."""
    return {
        "source_lang": "English",
        "target_lang": "Spanish",
        "source_text": "Hello world",
        "country": "Spain",
    }


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "AZURE_API_KEY": "test_azure_key",
        },
    ):
        yield


@pytest.fixture
def mock_httpx_client():
    """
    Mock httpx client to prevent any real HTTP requests during testing.
    This is a safety net in case any code tries to make direct HTTP calls.
    """
    with patch("httpx.Client") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_requests_session():
    """
    Mock requests session to prevent any real HTTP requests during testing.
    This is a safety net in case any code tries to make direct HTTP calls.
    """
    with patch("requests.Session") as mock_session:
        mock_instance = Mock()
        mock_session.return_value = mock_instance
        yield mock_instance
