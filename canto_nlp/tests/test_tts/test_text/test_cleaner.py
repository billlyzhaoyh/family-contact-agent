"""
Tests for the text cleaner module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from canto_nlp.tts.text.cleaner import (
    clean_text,
    clean_text_bert,
    text_to_sequence,
)


class TestCleanText:
    """Test the clean_text function."""

    @patch("canto_nlp.tts.text.cleaner.language_module_map")
    def test_clean_text_cantonese(self, mock_language_map, sample_cantonese_text):
        """Test Cantonese text cleaning."""
        # Mock the Cantonese module
        mock_cantonese_module = Mock()
        mock_cantonese_module.text_normalize.return_value = "normalized_text"
        mock_cantonese_module.g2p.return_value = (["n", "i", "h"], [1, 2, 3], [2, 1])
        mock_language_map.__getitem__.return_value = mock_cantonese_module

        # Call the function
        result = clean_text(sample_cantonese_text, "YUE")

        # Assertions
        expected_result = ("normalized_text", ["n", "i", "h"], [1, 2, 3], [2, 1])
        assert result == expected_result
        mock_cantonese_module.text_normalize.assert_called_once_with(
            sample_cantonese_text
        )
        mock_cantonese_module.g2p.assert_called_once_with("normalized_text")

    @patch("canto_nlp.tts.text.cleaner.language_module_map")
    def test_clean_text_english(self, mock_language_map, sample_english_text):
        """Test English text cleaning."""
        # Mock the English module
        mock_english_module = Mock()
        mock_english_module.text_normalize.return_value = "normalized_text"
        mock_english_module.g2p.return_value = (["h", "e", "l"], [1, 2, 3], [1, 1, 1])
        mock_language_map.__getitem__.return_value = mock_english_module

        # Call the function
        result = clean_text(sample_english_text, "EN")

        # Assertions
        expected_result = ("normalized_text", ["h", "e", "l"], [1, 2, 3], [1, 1, 1])
        assert result == expected_result
        mock_english_module.text_normalize.assert_called_once_with(sample_english_text)
        mock_english_module.g2p.assert_called_once_with("normalized_text")

    def test_clean_text_invalid_language(self):
        """Test clean_text with invalid language."""
        with pytest.raises(KeyError):
            clean_text("test text", "INVALID")


class TestCleanTextBert:
    """Test the clean_text_bert function."""

    @patch("canto_nlp.tts.text.cleaner.language_module_map")
    def test_clean_text_bert_cantonese(self, mock_language_map, sample_cantonese_text):
        """Test Cantonese text cleaning with BERT."""
        # Mock the Cantonese module
        mock_cantonese_module = Mock()
        mock_cantonese_module.text_normalize.return_value = "normalized_text"
        mock_cantonese_module.g2p.return_value = (["n", "i", "h"], [1, 2, 3], [2, 1])
        mock_cantonese_module.get_bert_feature.return_value = np.random.randn(1024, 3)
        mock_language_map.__getitem__.return_value = mock_cantonese_module

        # Call the function
        result = clean_text_bert(sample_cantonese_text, "YUE")

        # Assertions
        expected_phones = ["n", "i", "h"]
        expected_tones = [1, 2, 3]
        assert result[0] == expected_phones
        assert result[1] == expected_tones
        assert result[2].shape == (1024, 3)  # BERT features
        mock_cantonese_module.text_normalize.assert_called_once_with(
            sample_cantonese_text
        )
        mock_cantonese_module.g2p.assert_called_once_with("normalized_text")
        mock_cantonese_module.get_bert_feature.assert_called_once_with(
            "normalized_text", [2, 1]
        )

    @patch("canto_nlp.tts.text.cleaner.language_module_map")
    def test_clean_text_bert_english(self, mock_language_map, sample_english_text):
        """Test English text cleaning with BERT."""
        # Mock the English module
        mock_english_module = Mock()
        mock_english_module.text_normalize.return_value = "normalized_text"
        mock_english_module.g2p.return_value = (["h", "e", "l"], [1, 2, 3], [1, 1, 1])
        mock_english_module.get_bert_feature.return_value = np.random.randn(1024, 3)
        mock_language_map.__getitem__.return_value = mock_english_module

        # Call the function
        result = clean_text_bert(sample_english_text, "EN")

        # Assertions
        expected_phones = ["h", "e", "l"]
        expected_tones = [1, 2, 3]
        assert result[0] == expected_phones
        assert result[1] == expected_tones
        assert result[2].shape == (1024, 3)  # BERT features
        mock_english_module.text_normalize.assert_called_once_with(sample_english_text)
        mock_english_module.g2p.assert_called_once_with("normalized_text")
        mock_english_module.get_bert_feature.assert_called_once_with(
            "normalized_text", [1, 1, 1]
        )


class TestTextToSequence:
    """Test the text_to_sequence function."""

    @patch("canto_nlp.tts.text.cleaner.clean_text")
    @patch("canto_nlp.tts.text.cleaner.cleaned_text_to_sequence")
    def test_text_to_sequence_cantonese(
        self, mock_cleaned_to_seq, mock_clean_text, sample_cantonese_text
    ):
        """Test text to sequence conversion for Cantonese."""
        # Mock clean_text
        mock_clean_text.return_value = (
            "normalized_text",
            ["n", "i", "h"],
            [1, 2, 3],
            [2, 1],
        )

        # Mock cleaned_text_to_sequence
        mock_cleaned_to_seq.return_value = ([1, 2, 3], [4, 5, 6], [0, 0, 0])

        # Call the function
        result = text_to_sequence(sample_cantonese_text, "YUE")

        # Assertions
        expected_result = ([1, 2, 3], [4, 5, 6], [0, 0, 0])
        assert result == expected_result
        mock_clean_text.assert_called_once_with(sample_cantonese_text, "YUE")
        mock_cleaned_to_seq.assert_called_once_with(["n", "i", "h"], [1, 2, 3], "YUE")

    @patch("canto_nlp.tts.text.cleaner.clean_text")
    @patch("canto_nlp.tts.text.cleaner.cleaned_text_to_sequence")
    def test_text_to_sequence_english(
        self, mock_cleaned_to_seq, mock_clean_text, sample_english_text
    ):
        """Test text to sequence conversion for English."""
        # Mock clean_text
        mock_clean_text.return_value = (
            "normalized_text",
            ["h", "e", "l"],
            [1, 2, 3],
            [1, 1, 1],
        )

        # Mock cleaned_text_to_sequence
        mock_cleaned_to_seq.return_value = ([10, 11, 12], [13, 14, 15], [1, 1, 1])

        # Call the function
        result = text_to_sequence(sample_english_text, "EN")

        # Assertions
        expected_result = ([10, 11, 12], [13, 14, 15], [1, 1, 1])
        assert result == expected_result
        mock_clean_text.assert_called_once_with(sample_english_text, "EN")
        mock_cleaned_to_seq.assert_called_once_with(["h", "e", "l"], [1, 2, 3], "EN")
