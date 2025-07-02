"""
Tests for the TTS inference module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from canto_nlp.tts.infer import (
    clean_text,
    convert_pad_shape,
    sequence_mask,
    generate_path,
    OnnxInferenceSession,
)


class TestCleanText:
    """Test the clean_text function."""

    @patch("canto_nlp.tts.infer.language_module_map")
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

    @patch("canto_nlp.tts.infer.language_module_map")
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


class TestConvertPadShape:
    """Test the convert_pad_shape function."""

    def test_convert_pad_shape_simple(self):
        """Test simple pad shape conversion."""
        pad_shape = [[1, 2], [3, 4]]
        result = convert_pad_shape(pad_shape)
        expected = [3, 4, 1, 2]
        assert result == expected

    def test_convert_pad_shape_empty(self):
        """Test empty pad shape conversion."""
        pad_shape = []
        result = convert_pad_shape(pad_shape)
        expected = []
        assert result == expected

    def test_convert_pad_shape_single_element(self):
        """Test single element pad shape conversion."""
        pad_shape = [[1, 2]]
        result = convert_pad_shape(pad_shape)
        expected = [1, 2]
        assert result == expected


class TestSequenceMask:
    """Test the sequence_mask function."""

    def test_sequence_mask_basic(self):
        """Test basic sequence mask generation."""
        length = np.array([3, 5, 2])
        result = sequence_mask(length)

        expected = np.array(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, False, False, False],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_sequence_mask_with_max_length(self):
        """Test sequence mask with specified max length."""
        length = np.array([3, 5, 2])
        max_length = 4
        result = sequence_mask(length, max_length)

        expected = np.array(
            [
                [True, True, True, False],
                [True, True, True, True],
                [True, True, False, False],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_sequence_mask_single_element(self):
        """Test sequence mask with single element."""
        length = np.array([3])
        result = sequence_mask(length)

        expected = np.array([[True, True, True]])
        np.testing.assert_array_equal(result, expected)


class TestGeneratePath:
    """Test the generate_path function."""

    def test_generate_path_basic(self):
        """Test basic path generation."""
        duration = np.array([[[1, 2, 1]]])  # [b, 1, t_x]
        mask = np.array(
            [[[[True, True, True], [True, True, True], [True, True, True]]]]
        )  # [b, 1, t_y, t_x]

        result = generate_path(duration, mask)

        # Check shape
        assert result.shape == (1, 1, 3, 3)  # [b, 1, t_y, t_x]
        # Check that result is boolean
        assert result.dtype == bool

    def test_generate_path_simple(self):
        """Test simple path generation."""
        duration = np.array([[[1, 1]]])  # [b, 1, t_x]
        mask = np.array([[[[True, True], [True, True]]]])  # [b, 1, t_y, t_x]

        result = generate_path(duration, mask)

        # Check shape
        assert result.shape == (1, 1, 2, 2)
        # Check that result is boolean
        assert result.dtype == bool


class TestOnnxInferenceSession:
    """Test the OnnxInferenceSession class."""

    @patch("canto_nlp.tts.infer.ort.InferenceSession")
    def test_onnx_session_initialization(self, mock_inference_session):
        """Test ONNX session initialization."""
        # Mock the InferenceSession
        mock_session = Mock()
        mock_inference_session.return_value = mock_session

        # Test paths
        paths = {
            "enc": "path/to/enc.onnx",
            "emb_g": "path/to/emb_g.onnx",
            "dp": "path/to/dp.onnx",
            "sdp": "path/to/sdp.onnx",
            "flow": "path/to/flow.onnx",
            "dec": "path/to/dec.onnx",
        }

        # Create session
        session = OnnxInferenceSession(paths)

        # Assertions
        assert mock_inference_session.call_count == 6
        assert session.enc == mock_session
        assert session.emb_g == mock_session
        assert session.dp == mock_session
        assert session.sdp == mock_session
        assert session.flow == mock_session
        assert session.dec == mock_session

    @patch("canto_nlp.tts.infer.ort.InferenceSession")
    def test_onnx_session_call_basic(self, mock_inference_session, mock_onnx_session):
        """Test basic ONNX session call."""
        # Mock the InferenceSession to use the mock class from the fixture
        mock_inference_session.side_effect = mock_onnx_session

        # Create session
        paths = {
            "enc": "path/to/enc.onnx",
            "emb_g": "path/to/emb_g.onnx",
            "dp": "path/to/dp.onnx",
            "sdp": "path/to/sdp.onnx",
            "flow": "path/to/flow.onnx",
            "dec": "path/to/dec.onnx",
        }
        session = OnnxInferenceSession(paths)

        # Test inputs
        seq = np.array([1, 2, 3, 4, 5])
        tone = np.array([1, 2, 1, 2, 1])
        language = np.array([0, 0, 0, 0, 0])
        bert_en = np.random.randn(5, 1024)
        bert_yue = np.random.randn(5, 1024)
        sid = np.array([0])

        # Call the session
        session(seq, tone, language, bert_en, bert_yue, sid)

        # Check that the session was called 6 times (once for each ONNX model)
        assert mock_inference_session.call_count == 6

    @patch("canto_nlp.tts.infer.ort.InferenceSession")
    def test_onnx_session_call_with_parameters(
        self, mock_inference_session, mock_onnx_session
    ):
        """Test ONNX session call with different parameters."""
        # Mock the InferenceSession to use the mock class from the fixture
        mock_inference_session.side_effect = mock_onnx_session

        # Create session
        paths = {
            "enc": "path/to/enc.onnx",
            "emb_g": "path/to/emb_g.onnx",
            "dp": "path/to/dp.onnx",
            "sdp": "path/to/sdp.onnx",
            "flow": "path/to/flow.onnx",
            "dec": "path/to/dec.onnx",
        }
        session = OnnxInferenceSession(paths)

        # Test inputs
        seq = np.array([1, 2, 3])
        tone = np.array([1, 2, 1])
        language = np.array([0, 0, 0])
        bert_en = np.random.randn(3, 1024)
        bert_yue = np.random.randn(3, 1024)
        sid = np.array([0])

        # Call with custom parameters
        session(
            seq,
            tone,
            language,
            bert_en,
            bert_yue,
            sid,
            seed=12345,
            seq_noise_scale=0.5,
            sdp_noise_scale=0.3,
            length_scale=1.5,
            sdp_ratio=0.6,
        )

        # Check that the session was called 6 times (once for each ONNX model)
        assert mock_inference_session.call_count == 6

    @patch("canto_nlp.tts.infer.ort.InferenceSession")
    def test_onnx_session_input_validation(
        self, mock_inference_session, mock_onnx_session
    ):
        """Test ONNX session input validation."""
        # Mock the InferenceSession to use the mock class from the fixture
        mock_inference_session.side_effect = mock_onnx_session

        # Create session
        paths = {
            "enc": "path/to/enc.onnx",
            "emb_g": "path/to/emb_g.onnx",
            "dp": "path/to/dp.onnx",
            "sdp": "path/to/sdp.onnx",
            "flow": "path/to/flow.onnx",
            "dec": "path/to/dec.onnx",
        }
        session = OnnxInferenceSession(paths)

        # Test with 2D inputs (should work)
        seq = np.array([[1, 2, 3]])
        tone = np.array([[1, 2, 1]])
        language = np.array([[0, 0, 0]])
        bert_en = np.random.randn(3, 1024)
        bert_yue = np.random.randn(3, 1024)
        sid = np.array([0])

        # This should not raise an error
        session(seq, tone, language, bert_en, bert_yue, sid)

        # Check that the session was called 6 times (once for each ONNX model)
        assert mock_inference_session.call_count == 6

    @patch("canto_nlp.tts.infer.ort.InferenceSession")
    def test_onnx_session_invalid_inputs(
        self, mock_inference_session, mock_onnx_session
    ):
        """Test ONNX session with invalid inputs."""
        # Mock the InferenceSession to use the mock class from the fixture
        mock_inference_session.side_effect = mock_onnx_session

        # Create session
        paths = {
            "enc": "path/to/enc.onnx",
            "emb_g": "path/to/emb_g.onnx",
            "dp": "path/to/dp.onnx",
            "sdp": "path/to/sdp.onnx",
            "flow": "path/to/flow.onnx",
            "dec": "path/to/dec.onnx",
        }
        session = OnnxInferenceSession(paths)

        # Test with 3D inputs (should fail assertion)
        seq = np.array([[[1, 2, 3]]])
        tone = np.array([[[1, 2, 1]]])
        language = np.array([[[0, 0, 0]]])
        bert_en = np.random.randn(3, 1024)
        bert_yue = np.random.randn(3, 1024)
        sid = np.array([0])

        # This should raise an AssertionError
        with pytest.raises(AssertionError):
            session(seq, tone, language, bert_en, bert_yue, sid)
