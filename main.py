from pathlib import Path
from dotenv import load_dotenv
import argparse
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import os
from huggingface_hub import hf_hub_download

from translation_agent.libs.translation import Translation
from canto_nlp.tts.infer import OnnxInferenceSession
from canto_nlp.tts.text import cleaned_text_to_sequence, get_bert
from canto_nlp.tts.text.cleaner import clean_text
from whatsapp_mcp.whatsapp_mcp_server.whatsapp import (
    search_contacts,
    get_direct_chat_by_contact,
    list_messages_raw,
    download_media,
    send_audio_message,
)
from whatsapp_mcp.whatsapp_mcp_server.audio import convert_opus_ogg_to_mp3
from canto_nlp.asr.infer import transcribe_audio
import soundfile as sf
import sounddevice as sd

load_dotenv()

OnnxSession: Optional[OnnxInferenceSession] = None

MODELS: List[Dict[str, Union[str, List[str]]]] = [
    {
        "local_path": "./canto_nlp/tts/bert/bert-large-cantonese",
        "repo_id": "hon9kon9ize/bert-large-cantonese",
        "files": ["pytorch_model.bin"],
    },
    {
        "local_path": "./canto_nlp/tts/bert/deberta-v3-large",
        "repo_id": "microsoft/deberta-v3-large",
        "files": ["spm.model", "pytorch_model.bin"],
    },
    {
        "local_path": "./canto_nlp/tts/onnx",
        "repo_id": "hon9kon9ize/bert-vits-zoengjyutgaai-onnx",
        "files": [
            "BertVits2.2PT.json",
            "BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
            "BertVits2.2PT/BertVits2.2PT_emb.onnx",
            "BertVits2.2PT/BertVits2.2PT_dp.onnx",
            "BertVits2.2PT/BertVits2.2PT_sdp.onnx",
            "BertVits2.2PT/BertVits2.2PT_flow.onnx",
            "BertVits2.2PT/BertVits2.2PT_dec.onnx",
        ],
    },
]


def get_onnx_session() -> OnnxInferenceSession:
    global OnnxSession

    if OnnxSession is not None:
        return OnnxSession

    OnnxSession = OnnxInferenceSession(
        {
            "enc": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
            "emb_g": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
            "dp": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
            "sdp": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
            "flow": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
            "dec": "./canto_nlp/tts/onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
        },
        Providers=["CPUExecutionProvider"],
    )
    return OnnxSession


def download_model_files(repo_id: str, files: List[str], local_path: str) -> None:
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            hf_hub_download(
                repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
            )


def download_models() -> None:
    for data in MODELS:
        download_model_files(data["repo_id"], data["files"], data["local_path"])


def intersperse(lst: List[Any], item: Any) -> List[Any]:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(
    text: str,
    language_str: str,
    style_text: Optional[str] = None,
    style_weight: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    style_text = None if style_text == "" else style_text
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # add blank
    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1

    bert_ori = get_bert(
        norm_text, word2ph, language_str, "cpu", style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "EN":
        en_bert = bert_ori
        yue_bert = np.random.randn(1024, len(phone))
    elif language_str == "YUE":
        en_bert = np.random.randn(1024, len(phone))
        yue_bert = bert_ori
    else:
        raise ValueError("language_str should be EN or YUE")

    assert yue_bert.shape[-1] == len(
        phone
    ), f"Bert seq len {yue_bert.shape[-1]} != {len(phone)}"

    phone = np.asarray(phone)
    tone = np.asarray(tone)
    language = np.asarray(language)
    en_bert = np.asarray(en_bert.T)
    yue_bert = np.asarray(yue_bert.T)

    return en_bert, yue_bert, phone, tone, language


# Text-to-speech function
def text_to_speech(
    text: str, sid: int = 0, language: str = "YUE"
) -> Optional[np.ndarray]:
    Session = get_onnx_session()
    if not text.strip():
        return None
    en_bert, yue_bert, x, tone, language = get_text(text, language)
    sid = np.array([sid])
    audio = Session(x, tone, language, en_bert, yue_bert, sid, sdp_ratio=0.4)

    return audio[0][0]


def find_last_audio_message_from_other(messages: List[Any]) -> Optional[Any]:
    """
    Find the last audio message that's not from the current user.

    Args:
        messages: List of Message objects from list_messages()

    Returns:
        The last audio message from someone else, or None if not found
    """
    for message in messages:
        # Check if it's an audio message (has media_type and content contains [audio)
        if message.media_type == "audio" or (
            message.content and "[audio" in message.content
        ):
            # Check if it's not from me
            if not message.is_from_me:
                return message
    return None


def translate_english_to_cantonese(text: str) -> str:
    translation = Translation(
        source_text=text,
        source_lang="English",
        target_lang="Cantonese",
        country="Hong Kong",
    )
    translated_text = translation.translate()
    return translated_text


def translate_cantonese_to_english(text: str) -> str:
    translation = Translation(
        source_text=text,
        source_lang="Cantonese",
        target_lang="English",
        country="United Kingdom",
    )
    translated_text = translation.translate()
    return translated_text


def setup_models() -> None:
    """Initialize and download required models."""
    print("Current working directory:", os.getcwd())
    os.makedirs("./canto_nlp/tts/bert/bert-large-cantonese", exist_ok=True)
    os.makedirs("./canto_nlp/tts/bert/deberta-v3-large", exist_ok=True)
    os.makedirs("./canto_nlp/tts/onnx", exist_ok=True)
    assert os.path.exists("./canto_nlp/tts/bert/bert-large-cantonese")
    assert os.path.exists("./canto_nlp/tts/bert/deberta-v3-large")
    assert os.path.exists("./canto_nlp/tts/onnx")
    download_models()


def find_contact_by_phone(search_name: str, phone_number: str) -> Optional[Any]:
    """Find a contact by name and filter by phone number."""
    contacts = search_contacts(search_name)
    print(f"Contacts found: {contacts}")
    filtered_contacts = [c for c in contacts if c.phone_number == phone_number]
    return filtered_contacts[0] if filtered_contacts else None


def process_audio_message(
    audio_message: Any, chat_jid: str, play_audio: bool = True
) -> Optional[Dict[str, str]]:
    """Process an audio message: download, convert, play, and transcribe."""
    print(f"Processing audio message: {audio_message}")
    print(f"Message ID: {audio_message.id}")
    print(f"Timestamp: {audio_message.timestamp}")
    print(f"Content: {audio_message.content}")

    # Download the audio file
    audio_path = download_media(audio_message.id, chat_jid)
    print(f"Audio path: {audio_path}")

    if not audio_path:
        print("Failed to download audio file")
        return None

    # Convert Opus OGG to MP3 for better compatibility
    try:
        mp3_path = convert_opus_ogg_to_mp3(audio_path)

        if play_audio:
            play_audio_file(mp3_path)

        # Transcribe the audio
        return transcribe_and_translate(mp3_path)

    except Exception as e:
        print(f"Failed to convert to MP3: {e}")
        return None


def play_audio_file(mp3_path: str) -> None:
    """Play an audio file."""
    print(f"Converted to MP3: {mp3_path}")
    print(f"You can now play this MP3 file on any device: {mp3_path}")

    # Read the mp3 file and play it
    audio_data, sample_rate = sf.read(mp3_path)
    print(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")

    # Play the audio
    sd.play(audio_data, sample_rate)
    sd.wait()


def transcribe_and_translate(mp3_path: str) -> Optional[Dict[str, str]]:
    """Transcribe audio and translate the result."""
    try:
        print("Transcribing audio...")
        transcribed_text = transcribe_audio(mp3_path)

        if transcribed_text:
            print(f"Transcribed text: {transcribed_text}")
            # Translate text back to English
            translated_text = translate_cantonese_to_english(transcribed_text)
            print(f"Translated text: {translated_text}")
            return {"transcribed": transcribed_text, "translated": translated_text}
        else:
            print("No text was transcribed (audio might be silent or unclear)")
            return None

    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return None


def create_audio_file(text: str, output_path: Union[str, Path]) -> bool:
    """Create an audio file from text and optionally play it."""
    # Translate English to Cantonese
    translation = Translation(
        source_text=text,
        source_lang="English",
        target_lang="Cantonese",
        country="Hong Kong",
    )
    translated_text = translation.translate()
    print(f"Translated text: {translated_text}")

    # Generate audio
    audio = text_to_speech(translated_text, sid=0, language="YUE")

    if audio is not None:
        # Save audio file
        sf.write(output_path, audio, 44100)
        print(f"Audio saved as {output_path}")

        # Always play the audio for preview before asking for confirmation
        print("Playing audio preview...")
        sd.play(audio, samplerate=44100)
        sd.wait()

        return True
    else:
        print("No audio to play or save.")
        return False


def send_text_as_audio(text: str, contact_phone: str, contact_name: str) -> bool:
    """Send text as audio message to a contact."""
    # Find contact
    contact = find_contact_by_phone(contact_name, contact_phone)
    if not contact:
        print(f"Contact {contact_name} with phone {contact_phone} not found.")
        return False

    print(f"Contact JID: {contact.jid}")

    # Create output directory
    output_dir = Path("outputs") / contact.jid
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate unique filename
    unique_id = uuid.uuid4()
    output_path = output_dir / f"{unique_id}.wav"

    # Create audio file
    if create_audio_file(text, output_path):
        # Ask for confirmation before sending
        print(f"\nReady to send audio message to {contact_name} ({contact_phone})")
        print(f"Text: {text}")
        confirmation = input("Send message? (Y/N): ").strip().upper()

        if confirmation == "Y":
            # Send the audio file
            success, status_message = send_audio_message(contact.jid, str(output_path))
            if success:
                print(f"Audio message sent successfully: {status_message}")
                return True
            else:
                print(f"Failed to send audio message: {status_message}")
                return False
        else:
            print("Message sending cancelled.")
            return False
    else:
        return False


def receive_mode(contact_phone: str = "") -> bool:
    """Receive and process audio messages from a contact."""
    # Get chat and messages
    chat = get_direct_chat_by_contact(contact_phone)
    print(f"Chat: {chat}")
    messages = list_messages_raw(chat_jid=chat.jid)
    print(f"Messages: {messages}")

    # Find and process the last audio message from contact
    last_audio_message = find_last_audio_message_from_other(messages)
    if last_audio_message:
        result = process_audio_message(last_audio_message, chat.jid)
        if result:
            print("Audio processing completed successfully")
            return True
    else:
        print("No audio messages found from contact")
        return False


def main() -> None:
    """Main execution function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Family Contact Agent - Send/Receive Audio Messages"
    )
    parser.add_argument(
        "--mode",
        choices=["send", "receive", "interactive"],
        default="receive",
        help="Operation mode",
    )
    parser.add_argument("--text", type=str, help="Text to send (for send mode)")
    parser.add_argument("--phone", type=str, default="", help="Contact phone number")
    parser.add_argument(
        "--name", type=str, default="", help="Contact name to search for"
    )

    args = parser.parse_args()

    # Setup models (required for both modes)
    print("Setting up models...")
    setup_models()

    if args.mode == "send":
        if not args.text:
            print("Error: --text is required for send mode")
            return

        if not args.phone:
            print("Error: --phone is required for send mode")
            return

        if not args.name:
            print("Error: --name is required for send mode")
            return

        print(f"Sending message: {args.text}")
        success = send_text_as_audio(args.text, args.phone, args.name)
        if success:
            print("Send operation completed successfully")
        else:
            print("Send operation failed")

    elif args.mode == "receive":
        if not args.phone:
            print("Error: --phone is required for receive mode")
            return

        print("Receiving and processing audio messages...")
        success = receive_mode(args.phone)
        if success:
            print("Receive operation completed successfully")
        else:
            print("Receive operation failed")

    elif args.mode == "interactive":
        print("Interactive mode - choose an operation:")
        print("1. Send text as audio")
        print("2. Receive and process audio")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            text = input("Enter text to send: ").strip()
            if text:
                success = send_text_as_audio(text, args.phone, args.name)
                if success:
                    print("Send operation completed successfully")
                else:
                    print("Send operation failed")
            else:
                print("No text provided")

        elif choice == "2":
            success = receive_mode(args.phone)
            if success:
                print("Receive operation completed successfully")
            else:
                print("Receive operation failed")

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
