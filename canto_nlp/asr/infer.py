# Use a pipeline as a high-level helper
from transformers import pipeline
import soundfile as sf
from typing import Optional

# Global pipeline instance to avoid reloading the model
_pipe = None


def get_asr_pipeline():
    """Get or create the ASR pipeline instance."""
    global _pipe
    if _pipe is None:
        MODEL_NAME = "alvanlii/whisper-small-cantonese"
        device = "cpu"
        lang = "zh"

        _pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            device=device,
        )
        _pipe.model.config.forced_decoder_ids = _pipe.tokenizer.get_decoder_prompt_ids(
            language=lang, task="transcribe"
        )
    return _pipe


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file to text using Cantonese Whisper model.

    Args:
        audio_path (str): Path to the audio file (supports MP3, WAV, OGG, etc.)

    Returns:
        str: Transcribed text, or None if transcription failed

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        RuntimeError: If transcription fails
    """
    try:
        # Get the ASR pipeline
        pipe = get_asr_pipeline()

        # Transcribe the audio
        result = pipe(audio_path)
        transcribed_text = result["text"]

        return transcribed_text.strip() if transcribed_text else None

    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def transcribe_audio_with_info(audio_path: str) -> dict:
    """
    Transcribe audio file to text with additional information.

    Args:
        audio_path (str): Path to the audio file

    Returns:
        dict: Dictionary containing 'text' and 'metadata' keys
    """
    try:
        # Get audio file info
        audio_info = sf.info(audio_path)

        # Transcribe
        text = transcribe_audio(audio_path)

        return {
            "text": text,
            "metadata": {
                "file_path": audio_path,
                "duration": audio_info.duration,
                "sample_rate": audio_info.samplerate,
                "channels": audio_info.channels,
                "format": audio_info.format,
            },
        }

    except Exception as e:
        raise RuntimeError(f"Failed to transcribe audio with info: {str(e)}")


if __name__ == "__main__":
    # Example usage
    mp3_path = "path/to/your/audio/file.mp3"  # Replace with your audio file path

    try:
        # Simple transcription
        text = transcribe_audio(mp3_path)
        print(f"Transcribed text: {text}")

        # Transcription with metadata
        result = transcribe_audio_with_info(mp3_path)
        print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")
