"""
Tests for the Cantonese BERT module.
"""

import torch
from unittest.mock import Mock, patch

from canto_nlp.tts.text.cantonese_bert import get_bert_feature


class TestCantoneseBert:
    """Test the Cantonese BERT feature extraction."""

    @patch("canto_nlp.tts.text.cantonese_bert.AutoTokenizer")
    @patch("canto_nlp.tts.text.cantonese_bert.AutoModelForMaskedLM")
    def test_get_bert_feature_basic(self, mock_auto_model, mock_auto_tokenizer):
        """Test basic BERT feature extraction."""
        # Test inputs
        norm_text = "你好世界"
        number_of_phonemes = 6
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "token_type_ids": torch.tensor([[0] * 5]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock the model
        mock_model_instance = Mock()
        mock_hidden_states = [torch.randn(1, number_of_phonemes, 1024)] * 25
        # Create a mock return value that behaves like a dictionary with hidden_states
        mock_return = {"hidden_states": mock_hidden_states}
        mock_model_instance.return_value = mock_return
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_auto_model.from_pretrained.return_value = mock_model_instance

        word2ph = [2] * number_of_phonemes

        # Call the function
        result = get_bert_feature(norm_text, word2ph)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024  # Hidden dimension
        assert result.shape[1] == sum(word2ph)  # Total phonemes

        # Check that tokenizer and model were called
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_model.from_pretrained.assert_called_once()

    @patch("canto_nlp.tts.text.cantonese_bert.AutoTokenizer")
    @patch("canto_nlp.tts.text.cantonese_bert.AutoModelForMaskedLM")
    def test_get_bert_feature_with_style(self, mock_auto_model, mock_auto_tokenizer):
        """Test BERT feature extraction with style text."""
        # Test inputs
        norm_text = "你好世界"
        number_of_phonemes = 6
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "token_type_ids": torch.tensor([[0] * 5]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock the model
        mock_model_instance = Mock()
        mock_hidden_states = [torch.randn(1, number_of_phonemes, 1024)] * 25
        # Create a mock return value that behaves like a dictionary with hidden_states
        mock_return = {"hidden_states": mock_hidden_states}
        mock_model_instance.return_value = mock_return
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_auto_model.from_pretrained.return_value = mock_model_instance

        word2ph = [2] * number_of_phonemes
        style_text = "快樂的語調"
        style_weight = 0.8

        # Call the function
        result = get_bert_feature(
            norm_text, word2ph, style_text=style_text, style_weight=style_weight
        )

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)

    @patch("canto_nlp.tts.text.cantonese_bert.AutoTokenizer")
    @patch("canto_nlp.tts.text.cantonese_bert.AutoModelForMaskedLM")
    def test_get_bert_feature_long_text(self, mock_auto_model, mock_auto_tokenizer):
        """Test BERT feature extraction with long text."""
        # Test inputs
        norm_text = "這是一個很長的句子,用來測試BERT特徵提取的處理能力。" * 10
        number_of_phonemes = 282
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1] + [i for i in range(2, 100)]]),
            "token_type_ids": torch.tensor([[0] * 99]),
            "attention_mask": torch.tensor([[1] * 99]),
        }
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock the model
        mock_model_instance = Mock()
        mock_hidden_states = [torch.randn(1, number_of_phonemes, 1024)] * 25
        # Create a mock return value that behaves like a dictionary with hidden_states
        mock_return = {"hidden_states": mock_hidden_states}
        mock_model_instance.return_value = mock_return
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_auto_model.from_pretrained.return_value = mock_model_instance

        word2ph = [2] * number_of_phonemes  # 282 phonemes total

        # Call the function
        result = get_bert_feature(norm_text, word2ph)

        # Assertions
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1024
        assert result.shape[1] == sum(word2ph)
