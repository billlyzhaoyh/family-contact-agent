"""
Common test fixtures and configuration for canto_nlp tests.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import soundfile as sf


@pytest.fixture
def sample_cantonese_text():
    """Sample Cantonese text for testing."""
    return "你好，世界！"


@pytest.fixture
def sample_english_text():
    """Sample English text for testing."""
    return "Hello, world!"


@pytest.fixture
def sample_cantonese_sentence():
    """Longer Cantonese sentence for testing."""
    return "我今日去咗超市買嘢，買咗好多生果同蔬菜。"


@pytest.fixture
def sample_english_sentence():
    """Longer English sentence for testing."""
    return "I went to the supermarket today and bought lots of fruits and vegetables."


@pytest.fixture
def mock_audio_file():
    """Create a temporary mock audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a simple sine wave audio data
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        # Save as WAV file
        sf.write(f.name, audio_data, sample_rate)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def mock_cantonese_audio_file():
    """Create a temporary mock Cantonese audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a simple sine wave audio data
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        # Save as WAV file
        sf.write(f.name, audio_data, sample_rate)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def mock_english_audio_file():
    """Create a temporary mock English audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a simple sine wave audio data
        sample_rate = 16000
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 880 * t)  # 880 Hz sine wave
        # Save as WAV file
        sf.write(f.name, audio_data, sample_rate)
        yield f.name
        os.unlink(f.name)


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX inference session."""
    # Create individual mock sessions for each ONNX model
    mock_emb_g = Mock()
    mock_emb_g.run.return_value = [np.random.randn(1, 512)]

    mock_enc = Mock()
    mock_enc.run.return_value = [
        np.random.randn(1, 80, 100),  # x
        np.random.randn(1, 80, 100),  # m_p
        np.random.randn(1, 80, 100),  # logs_p
        np.random.randn(1, 1, 100),  # x_mask
    ]

    mock_sdp = Mock()
    mock_sdp.run.return_value = [np.random.randn(1, 1, 100)]

    mock_dp = Mock()
    mock_dp.run.return_value = [np.random.randn(1, 1, 100)]

    mock_flow = Mock()
    mock_flow.run.return_value = [np.random.randn(1, 80, 200)]

    mock_dec = Mock()
    mock_dec.run.return_value = [np.random.randn(1, 1, 200 * 256)]  # Audio output

    # Create a mock InferenceSession class that returns the appropriate mock based on the path
    class MockInferenceSession:
        def __init__(self, path, providers=None):
            if "emb_g" in path:
                self.run = mock_emb_g.run
            elif "enc" in path:
                self.run = mock_enc.run
            elif "sdp" in path:
                self.run = mock_sdp.run
            elif "dp" in path:
                self.run = mock_dp.run
            elif "flow" in path:
                self.run = mock_flow.run
            elif "dec" in path:
                self.run = mock_dec.run
            else:
                self.run = Mock().run

    return MockInferenceSession


@pytest.fixture
def mock_whisper_pipeline():
    """Mock Whisper ASR pipeline."""
    mock_pipe = Mock()
    mock_pipe.return_value = {"text": "Mock transcribed text"}
    return mock_pipe


@pytest.fixture
def mock_whisper_pipeline_cantonese():
    """Mock Whisper ASR pipeline for Cantonese."""
    mock_pipe = Mock()
    mock_pipe.return_value = {"text": "你好世界"}
    return mock_pipe


@pytest.fixture
def mock_whisper_pipeline_english():
    """Mock Whisper ASR pipeline for English."""
    mock_pipe = Mock()
    mock_pipe.return_value = {"text": "Hello world"}
    return mock_pipe


@pytest.fixture
def mock_audio_info():
    """Mock audio file information."""
    mock_info = Mock()
    mock_info.duration = 2.5
    mock_info.samplerate = 16000
    mock_info.channels = 1
    mock_info.format = "WAV"
    return mock_info


@pytest.fixture
def sample_phone_sequence():
    """Sample phone sequence for testing."""
    return ["n", "i", "h", "a", "o"]


@pytest.fixture
def sample_tone_sequence():
    """Sample tone sequence for testing."""
    return [2, 5, 3]


@pytest.fixture
def sample_word2ph():
    """Sample word-to-phoneme mapping for testing."""
    return [2, 3, 2]


@pytest.fixture
def mock_bert_features():
    """Mock BERT features for testing."""
    return np.random.randn(1024, 10)  # 1024 dim features, 10 time steps


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "TRANSFORMERS_CACHE": "/tmp/test_cache",
            "HF_HOME": "/tmp/test_hf_home",
        },
    ):
        yield
