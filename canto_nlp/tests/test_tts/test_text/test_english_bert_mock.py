"""
Tests for the English BERT mock module.
"""

import torch

from canto_nlp.tts.text.english_bert_mock import get_bert_feature


class TestEnglishBertMock:
    """Test the English BERT mock feature extraction."""

    def test_get_bert_feature_basic(self):
        """Test basic BERT feature extraction."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]  # Word to phoneme mapping

        # Call the function
        result = get_bert_feature(norm_text, word2ph)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024  # Hidden dimension
        assert result.shape[1] == sum(word2ph)  # Total phonemes

    def test_get_bert_feature_with_style(self):
        """Test BERT feature extraction with style text."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]
        style_text = "Happy tone"
        style_weight = 0.8

        # Call the function
        result = get_bert_feature(
            norm_text, word2ph, style_text=style_text, style_weight=style_weight
        )

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)

    def test_get_bert_feature_long_text(self):
        """Test BERT feature extraction with long text."""
        # Test inputs
        norm_text = (
            "This is a very long sentence for testing BERT feature extraction capabilities."
            * 10
        )
        word2ph = [2] * 142  # 142 phonemes total

        # Call the function
        result = get_bert_feature(norm_text, word2ph)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)

    def test_get_bert_feature_consistency(self):
        """Test that BERT features are consistent for same inputs."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]

        # Call the function multiple times
        result1 = get_bert_feature(norm_text, word2ph)
        result2 = get_bert_feature(norm_text, word2ph)

        # Assertions
        assert isinstance(result1, torch.Tensor)
        assert isinstance(result2, torch.Tensor)
        assert result1.shape == result2.shape
        # Note: Since this is a mock implementation, we don't expect exact same values
        # but we do expect consistent shapes and types

    def test_get_bert_feature_style_weight_range(self):
        """Test BERT feature extraction with different style weights."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]
        style_text = "Happy tone"

        # Test different style weights
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = get_bert_feature(
                norm_text, word2ph, style_text=style_text, style_weight=weight
            )
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == 1024
            assert result.shape[1] == sum(word2ph)

    def test_get_bert_feature_no_style_text(self):
        """Test BERT feature extraction without style text."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]

        # Call without style text
        result = get_bert_feature(norm_text, word2ph, style_text=None, style_weight=0.7)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)

    def test_get_bert_feature_empty_style_text(self):
        """Test BERT feature extraction with empty style text."""
        # Test inputs
        norm_text = "Hello world"
        word2ph = [2, 2, 1, 1]

        # Call with empty style text
        result = get_bert_feature(norm_text, word2ph, style_text="", style_weight=0.7)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)
