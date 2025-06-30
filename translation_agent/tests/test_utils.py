"""
Unit tests for translation_agent.libs.utils module.
"""

import pytest
from unittest.mock import patch, Mock

from translation_agent.libs.utils import (
    split_text_by_token_size,
    num_tokens_in_string,
    calculate_chunk_size,
    MAX_TOKENS_PER_CHUNK,
)


class TestUtils:
    """Test cases for utility functions."""

    def test_max_tokens_per_chunk_constant(self):
        """Test that MAX_TOKENS_PER_CHUNK has the expected value."""
        assert MAX_TOKENS_PER_CHUNK == 500

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello world", 2),  # Simple English text
            ("", 0),  # Empty string
            ("你好世界", 5),  # Chinese characters
            ("Hello 你好", 4),  # Mixed languages
            ("This is a longer sentence with more words.", 9),  # Longer text
        ],
    )
    def test_num_tokens_in_string(self, input_str, expected):
        """Test token counting for various input strings."""
        result = num_tokens_in_string(input_str)
        assert result == expected

    def test_num_tokens_in_string_custom_encoding(self):
        """Test token counting with custom encoding."""
        result = num_tokens_in_string("Hello world", encoding_name="cl100k_base")
        assert result == 2

    def test_num_tokens_in_string_invalid_encoding(self):
        """Test token counting with invalid encoding raises error."""
        with pytest.raises(
            Exception
        ):  # tiktoken will raise an error for invalid encoding
            num_tokens_in_string("Hello world", encoding_name="invalid_encoding")

    @pytest.mark.parametrize(
        "token_count,token_limit,expected",
        [
            (100, 500, 100),  # Text fits in one chunk
            (500, 500, 500),  # Text exactly fits in one chunk
            (1000, 500, 500),  # Text needs to be split into 2 chunks
            (1500, 500, 500),  # Text needs to be split into 3 chunks
            (0, 500, 0),  # Empty text
            (1, 500, 1),  # Single token
        ],
    )
    def test_calculate_chunk_size(self, token_count, token_limit, expected):
        """Test chunk size calculation for various scenarios."""
        result = calculate_chunk_size(token_count, token_limit)
        assert result == expected

    def test_calculate_chunk_size_with_remainder(self):
        """Test chunk size calculation when there's a remainder."""
        # 1200 tokens with 500 limit = 3 chunks (466, 466, 466)
        result = calculate_chunk_size(1200, 500)
        assert result == 466

    def test_calculate_chunk_size_negative_values(self):
        """Test chunk size calculation with negative values."""
        result = calculate_chunk_size(-100, 500)
        assert result == -100  # Should return original token count

    @patch("translation_agent.libs.utils.num_tokens_in_string")
    @patch("translation_agent.libs.utils.calculate_chunk_size")
    @patch("translation_agent.libs.utils.RecursiveCharacterTextSplitter")
    def test_split_text_by_token_size_short_text(
        self, mock_splitter_class, mock_calculate_chunk_size, mock_num_tokens
    ):
        """Test text splitting for short text that doesn't need splitting."""
        # Mock the token count to be less than limit
        mock_num_tokens.return_value = 100

        # Mock the chunk size calculation
        mock_calculate_chunk_size.return_value = 100

        # Mock the text splitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Short text"]
        mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

        result = split_text_by_token_size("Short text")

        # Verify the function calls
        mock_num_tokens.assert_called_once_with("Short text")
        mock_calculate_chunk_size.assert_called_once_with(
            token_count=100, token_limit=500
        )
        mock_splitter_class.from_tiktoken_encoder.assert_called_once_with(
            model_name="gpt-4", chunk_size=100, chunk_overlap=0
        )
        mock_splitter.split_text.assert_called_once_with("Short text")

        assert result == ["Short text"]

    @patch("translation_agent.libs.utils.num_tokens_in_string")
    @patch("translation_agent.libs.utils.calculate_chunk_size")
    @patch("translation_agent.libs.utils.RecursiveCharacterTextSplitter")
    def test_split_text_by_token_size_long_text(
        self, mock_splitter_class, mock_calculate_chunk_size, mock_num_tokens
    ):
        """Test text splitting for long text that needs splitting."""
        # Mock the token count to be more than limit
        mock_num_tokens.return_value = 1000

        # Mock the chunk size calculation
        mock_calculate_chunk_size.return_value = 500

        # Mock the text splitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

        result = split_text_by_token_size("Long text that needs splitting")

        # Verify the function calls
        mock_num_tokens.assert_called_once_with("Long text that needs splitting")
        mock_calculate_chunk_size.assert_called_once_with(
            token_count=1000, token_limit=500
        )
        mock_splitter_class.from_tiktoken_encoder.assert_called_once_with(
            model_name="gpt-4", chunk_size=500, chunk_overlap=0
        )
        mock_splitter.split_text.assert_called_once_with(
            "Long text that needs splitting"
        )

        assert result == ["Chunk 1", "Chunk 2"]

    @patch("translation_agent.libs.utils.num_tokens_in_string")
    @patch("translation_agent.libs.utils.calculate_chunk_size")
    @patch("translation_agent.libs.utils.RecursiveCharacterTextSplitter")
    def test_split_text_by_token_size_custom_limit(
        self, mock_splitter_class, mock_calculate_chunk_size, mock_num_tokens
    ):
        """Test text splitting with custom token limit."""
        mock_num_tokens.return_value = 800
        mock_calculate_chunk_size.return_value = 400
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

        result = split_text_by_token_size("Test text", token_limit=300)

        # Verify custom token limit is used
        mock_calculate_chunk_size.assert_called_once_with(
            token_count=800, token_limit=300
        )
        mock_splitter_class.from_tiktoken_encoder.assert_called_once_with(
            model_name="gpt-4", chunk_size=400, chunk_overlap=0
        )

        assert result == ["Chunk 1", "Chunk 2"]

    def test_split_text_by_token_size_empty_text(self):
        """Test text splitting with empty text."""
        result = split_text_by_token_size("")
        assert (
            result == []
        )  # RecursiveCharacterTextSplitter returns list with no chunks

    def test_split_text_by_token_size_whitespace_only(self):
        """Test text splitting with whitespace-only text."""
        result = split_text_by_token_size("   \n\t   ")
        assert (
            result == []
        )  # RecursiveCharacterTextSplitter returns list with no chunks

    @patch("translation_agent.libs.utils.num_tokens_in_string")
    def test_split_text_by_token_size_unicode_text(self, mock_num_tokens):
        """Test text splitting with Unicode text."""
        mock_num_tokens.return_value = 100

        with patch("translation_agent.libs.utils.calculate_chunk_size") as mock_calc:
            mock_calc.return_value = 100
            with patch(
                "translation_agent.libs.utils.RecursiveCharacterTextSplitter"
            ) as mock_splitter_class:
                mock_splitter = Mock()
                mock_splitter.split_text.return_value = ["Unicode text: 你好世界"]
                mock_splitter_class.from_tiktoken_encoder.return_value = mock_splitter

                result = split_text_by_token_size("Unicode text: 你好世界")

                assert result == ["Unicode text: 你好世界"]
                mock_num_tokens.assert_called_once_with("Unicode text: 你好世界")
