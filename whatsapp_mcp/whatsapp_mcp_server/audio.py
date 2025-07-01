import os
import subprocess
import tempfile


def convert_to_opus_ogg(input_file, output_file=None, bitrate="32k", sample_rate=24000):
    """
    Convert an audio file to Opus format in an Ogg container.

    Args:
        input_file (str): Path to the input audio file
        output_file (str, optional): Path to save the output file. If None, replaces the
                                    extension of input_file with .ogg
        bitrate (str, optional): Target bitrate for Opus encoding (default: "32k")
        sample_rate (int, optional): Sample rate for output (default: 24000)

    Returns:
        str: Path to the converted file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # If no output file is specified, replace the extension with .ogg
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".ogg"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the ffmpeg command
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-ar",
        str(sample_rate),
        "-application",
        "voip",  # Optimize for voice
        "-vbr",
        "on",  # Variable bitrate
        "-compression_level",
        "10",  # Maximum compression
        "-frame_duration",
        "60",  # 60ms frames (good for voice)
        "-y",  # Overwrite output file if it exists
        output_file,
    ]

    try:
        # Run the ffmpeg command and capture output
        subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert audio. You likely need to install ffmpeg {e.stderr}"
        )


def convert_to_opus_ogg_temp(input_file, bitrate="32k", sample_rate=24000):
    """
    Convert an audio file to Opus format in an Ogg container and store in a temporary file.

    Args:
        input_file (str): Path to the input audio file
        bitrate (str, optional): Target bitrate for Opus encoding (default: "32k")
        sample_rate (int, optional): Sample rate for output (default: 24000)

    Returns:
        str: Path to the temporary file with the converted audio

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    # Create a temporary file with .ogg extension
    temp_file = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    temp_file.close()

    try:
        # Convert the audio
        convert_to_opus_ogg(input_file, temp_file.name, bitrate, sample_rate)
        return temp_file.name
    except Exception as e:
        # Clean up the temporary file if conversion fails
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


def convert_opus_ogg_to_wav(input_file, output_file=None, sample_rate=44100):
    """
    Convert an Opus OGG file to WAV format for playback on Mac.

    Args:
        input_file (str): Path to the input Opus OGG file
        output_file (str, optional): Path to save the output WAV file. If None, replaces the
                                    extension of input_file with .wav
        sample_rate (int, optional): Sample rate for output (default: 44100)

    Returns:
        str: Path to the converted WAV file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # If no output file is specified, replace the extension with .wav
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the ffmpeg command to convert Opus OGG to WAV
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:a",
        "pcm_s16le",  # 16-bit PCM audio
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        "1",  # Mono audio (typical for voice messages)
        "-y",  # Overwrite output file if it exists
        output_file,
    ]

    try:
        # Run the ffmpeg command and capture output
        subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert Opus OGG to WAV. You likely need to install ffmpeg: {e.stderr}"
        )


def convert_opus_ogg_to_wav_temp(input_file, sample_rate=44100):
    """
    Convert an Opus OGG file to WAV format and store in a temporary file.

    Args:
        input_file (str): Path to the input Opus OGG file
        sample_rate (int, optional): Sample rate for output (default: 44100)

    Returns:
        str: Path to the temporary WAV file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    # Create a temporary file with .wav extension
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    try:
        # Convert the audio
        convert_opus_ogg_to_wav(input_file, temp_file.name, sample_rate)
        return temp_file.name
    except Exception as e:
        # Clean up the temporary file if conversion fails
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e


def convert_opus_ogg_to_mp3(input_file, output_file=None, bitrate="128k"):
    """
    Convert an Opus OGG file to MP3 format for better compatibility.

    Args:
        input_file (str): Path to the input Opus OGG file
        output_file (str, optional): Path to save the output MP3 file. If None, replaces the
                                    extension of input_file with .mp3
        bitrate (str, optional): MP3 bitrate (default: "128k")

    Returns:
        str: Path to the converted MP3 file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # If no output file is specified, replace the extension with .mp3
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".mp3"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the ffmpeg command to convert Opus OGG to MP3
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-c:a",
        "libmp3lame",  # MP3 encoder
        "-b:a",
        bitrate,  # Bitrate
        "-ar",
        "44100",  # Sample rate
        "-ac",
        "1",  # Mono audio (typical for voice messages)
        "-y",  # Overwrite output file if it exists
        output_file,
    ]

    try:
        # Run the ffmpeg command and capture output
        subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert Opus OGG to MP3. You likely need to install ffmpeg: {e.stderr}"
        )


def convert_opus_ogg_to_mp3_temp(input_file, bitrate="128k"):
    """
    Convert an Opus OGG file to MP3 format and store in a temporary file.

    Args:
        input_file (str): Path to the input Opus OGG file
        bitrate (str, optional): MP3 bitrate (default: "128k")

    Returns:
        str: Path to the temporary MP3 file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        RuntimeError: If the ffmpeg conversion fails
    """
    # Create a temporary file with .mp3 extension
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_file.close()

    try:
        # Convert the audio
        convert_opus_ogg_to_mp3(input_file, temp_file.name, bitrate)
        return temp_file.name
    except Exception as e:
        # Clean up the temporary file if conversion fails
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise e
