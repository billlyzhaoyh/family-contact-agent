"""
Tests for the ASR inference module.
"""

import pytest
from unittest.mock import Mock, patch

from canto_nlp.asr.infer import (
    get_asr_pipeline,
    transcribe_audio,
    transcribe_audio_with_info,
)


class TestGetAsrPipeline:
    """Test the ASR pipeline initialization."""

    @patch("canto_nlp.asr.infer._pipe", None)
    @patch("canto_nlp.asr.infer.pipeline")
    def test_get_asr_pipeline_initialization(self, mock_pipeline):
        """Test that the ASR pipeline is initialized correctly."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        # Mock the tokenizer and model config
        mock_pipe.tokenizer = Mock()
        mock_pipe.model = Mock()
        mock_pipe.model.config = Mock()
        mock_pipe.tokenizer.get_decoder_prompt_ids.return_value = [1, 2, 3]

        # Call the function
        result = get_asr_pipeline()

        # Assertions
        assert result == mock_pipe
        mock_pipeline.assert_called_once_with(
            task="automatic-speech-recognition",
            model="alvanlii/whisper-small-cantonese",
            chunk_length_s=30,
            device="cpu",
        )
        mock_pipe.tokenizer.get_decoder_prompt_ids.assert_called_once_with(
            language="zh", task="transcribe"
        )
        assert mock_pipe.model.config.forced_decoder_ids == [1, 2, 3]

    def test_get_asr_pipeline_singleton(self):
        """Test that the pipeline follows singleton pattern."""
        # Reset the global pipeline to None
        import canto_nlp.asr.infer as asr_module

        asr_module._pipe = None

        # First call should create a new pipeline
        with patch("canto_nlp.asr.infer.pipeline") as mock_pipeline:
            mock_pipe = Mock()
            mock_pipeline.return_value = mock_pipe
            mock_pipe.tokenizer = Mock()
            mock_pipe.model = Mock()
            mock_pipe.model.config = Mock()
            mock_pipe.tokenizer.get_decoder_prompt_ids.return_value = [1, 2, 3]

            result1 = get_asr_pipeline()

            # Second call should return the same pipeline
            result2 = get_asr_pipeline()

            # Both should be the same object (singleton)
            assert result1 is result2
            assert result1 == mock_pipe
            # Pipeline should only be called once
            mock_pipeline.assert_called_once()


class TestTranscribeAudio:
    """Test the transcribe_audio function."""

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_success(self, mock_get_pipeline, mock_audio_file):
        """Test successful audio transcription."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = {"text": "你好世界"}
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        result = transcribe_audio(mock_audio_file)

        # Assertions
        assert result == "你好世界"
        mock_pipe.assert_called_once_with(mock_audio_file)

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_empty_result(self, mock_get_pipeline, mock_audio_file):
        """Test transcription with empty result."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = {"text": ""}
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        result = transcribe_audio(mock_audio_file)

        # Assertions
        assert result is None

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_whitespace_result(
        self, mock_get_pipeline, mock_audio_file
    ):
        """Test transcription with whitespace-only result."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = {"text": "   \n\t   "}
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        result = transcribe_audio(mock_audio_file)

        # Assertions
        assert result == ""

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_file_not_found(self, mock_get_pipeline):
        """Test transcription with non-existent file."""
        # Mock the pipeline to raise FileNotFoundError
        mock_pipe = Mock()
        mock_pipe.side_effect = FileNotFoundError(
            "Audio file not found: /path/to/non/existent/file.wav"
        )
        mock_get_pipeline.return_value = mock_pipe

        non_existent_file = "/path/to/non/existent/file.wav"

        with pytest.raises(
            FileNotFoundError, match=f"Audio file not found: {non_existent_file}"
        ):
            transcribe_audio(non_existent_file)

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_pipeline_error(self, mock_get_pipeline, mock_audio_file):
        """Test transcription when pipeline raises an error."""
        # Mock the pipeline to raise an error
        mock_pipe = Mock()
        mock_pipe.side_effect = Exception("Pipeline error")
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        with pytest.raises(RuntimeError, match="Transcription failed: Pipeline error"):
            transcribe_audio(mock_audio_file)

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_cantonese(
        self, mock_get_pipeline, mock_cantonese_audio_file
    ):
        """Test Cantonese audio transcription."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = {"text": "我今日去咗超市買嘢"}
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        result = transcribe_audio(mock_cantonese_audio_file)

        # Assertions
        assert result == "我今日去咗超市買嘢"
        mock_pipe.assert_called_once_with(mock_cantonese_audio_file)

    @patch("canto_nlp.asr.infer.get_asr_pipeline")
    def test_transcribe_audio_english(self, mock_get_pipeline, mock_english_audio_file):
        """Test English audio transcription."""
        # Mock the pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = {"text": "Hello world, how are you today?"}
        mock_get_pipeline.return_value = mock_pipe

        # Call the function
        result = transcribe_audio(mock_english_audio_file)

        # Assertions
        assert result == "Hello world, how are you today?"
        mock_pipe.assert_called_once_with(mock_english_audio_file)


class TestTranscribeAudioWithInfo:
    """Test the transcribe_audio_with_info function."""

    @patch("canto_nlp.asr.infer.sf.info")
    @patch("canto_nlp.asr.infer.transcribe_audio")
    def test_transcribe_audio_with_info_success(
        self, mock_transcribe, mock_sf_info, mock_audio_file, mock_audio_info
    ):
        """Test successful transcription with metadata."""
        # Mock dependencies
        mock_transcribe.return_value = "你好世界"
        mock_sf_info.return_value = mock_audio_info

        # Call the function
        result = transcribe_audio_with_info(mock_audio_file)

        # Assertions
        expected_result = {
            "text": "你好世界",
            "metadata": {
                "file_path": mock_audio_file,
                "duration": 2.5,
                "sample_rate": 16000,
                "channels": 1,
                "format": "WAV",
            },
        }
        assert result == expected_result
        mock_transcribe.assert_called_once_with(mock_audio_file)
        mock_sf_info.assert_called_once_with(mock_audio_file)

    @patch("canto_nlp.asr.infer.sf.info")
    @patch("canto_nlp.asr.infer.transcribe_audio")
    def test_transcribe_audio_with_info_transcription_error(
        self, mock_transcribe, mock_sf_info, mock_audio_file
    ):
        """Test transcription with info when transcription fails."""
        # Mock transcription to raise an error
        mock_transcribe.side_effect = RuntimeError("Transcription failed")

        # Call the function
        with pytest.raises(
            RuntimeError,
            match="Failed to transcribe audio with info: Transcription failed",
        ):
            transcribe_audio_with_info(mock_audio_file)

    @patch("canto_nlp.asr.infer.sf.info")
    @patch("canto_nlp.asr.infer.transcribe_audio")
    def test_transcribe_audio_with_info_sf_error(
        self, mock_transcribe, mock_sf_info, mock_audio_file
    ):
        """Test transcription with info when soundfile.info fails."""
        # Mock soundfile.info to raise an error
        mock_sf_info.side_effect = Exception("Soundfile error")

        # Call the function
        with pytest.raises(
            RuntimeError, match="Failed to transcribe audio with info: Soundfile error"
        ):
            transcribe_audio_with_info(mock_audio_file)

    @patch("canto_nlp.asr.infer.sf.info")
    @patch("canto_nlp.asr.infer.transcribe_audio")
    def test_transcribe_audio_with_info_english(
        self, mock_transcribe, mock_sf_info, mock_english_audio_file, mock_audio_info
    ):
        """Test English audio transcription with info."""
        # Mock dependencies
        mock_transcribe.return_value = "Hello world, how are you today?"
        mock_sf_info.return_value = mock_audio_info

        # Call the function
        result = transcribe_audio_with_info(mock_english_audio_file)

        # Assertions
        expected_result = {
            "text": "Hello world, how are you today?",
            "metadata": {
                "file_path": mock_english_audio_file,
                "duration": 2.5,
                "sample_rate": 16000,
                "channels": 1,
                "format": "WAV",
            },
        }
        assert result == expected_result
