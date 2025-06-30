from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import os
from huggingface_hub import hf_hub_download

from canto_nlp.tts.infer import OnnxInferenceSession
from canto_nlp.tts.text import cleaned_text_to_sequence, get_bert
from canto_nlp.tts.text.cleaner import clean_text
from whatsapp_mcp.whatsapp_mcp_server.whatsapp import (
    search_contacts,
    get_direct_chat_by_contact,
    list_messages_raw,
    download_media,
)
from whatsapp_mcp.whatsapp_mcp_server.audio import convert_opus_ogg_to_mp3
from canto_nlp.asr.infer import transcribe_audio
import soundfile as sf
import sounddevice as sd

load_dotenv()

OnnxSession = None

MODELS = [
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


def get_onnx_session():
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


def download_model_files(repo_id, files, local_path):
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            hf_hub_download(
                repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
            )


def download_models():
    for data in MODELS:
        download_model_files(data["repo_id"], data["files"], data["local_path"])


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(text, language_str, style_text=None, style_weight=0.7):
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
def text_to_speech(text, sid=0, language="YUE"):
    Session = get_onnx_session()
    if not text.strip():
        return None
    en_bert, yue_bert, x, tone, language = get_text(text, language)
    sid = np.array([sid])
    audio = Session(x, tone, language, en_bert, yue_bert, sid, sdp_ratio=0.4)

    return audio[0][0]


def find_last_audio_message_from_other(messages):
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


if __name__ == "__main__":
    # Ensure all required directories exist before downloading models
    print("Current working directory:", os.getcwd())
    os.makedirs("./canto_nlp/tts/bert/bert-large-cantonese", exist_ok=True)
    os.makedirs("./canto_nlp/tts/bert/deberta-v3-large", exist_ok=True)
    os.makedirs("./canto_nlp/tts/onnx", exist_ok=True)
    assert os.path.exists("./canto_nlp/tts/bert/bert-large-cantonese")
    assert os.path.exists("./canto_nlp/tts/bert/deberta-v3-large")
    assert os.path.exists("./canto_nlp/tts/onnx")
    download_models()
    text_to_baba = "Is it okay to have lunch at Laksa for 9 people next Wendesday at 12:30pm? Can we have the room downstairs? Thanks!"
    contacts = search_contacts("Michael")
    print(f"Contacts found: {contacts}")
    # filter for phonenumber 447711957486:
    filtered_contacts = [c for c in contacts if c.phone_number == "447711957486"]
    baba_jid = filtered_contacts[0].jid if filtered_contacts else None
    print(f"Baba's JID: {baba_jid}")
    if baba_jid:
        chat = get_direct_chat_by_contact("447711957486")
        print(f"Chat: {chat}")
        messages = list_messages_raw(chat_jid=chat.jid)
        print(f"Messages: {messages}")
        # Find the last audio message from Baba
        last_audio_from_baba = find_last_audio_message_from_other(messages)
        if last_audio_from_baba:
            print(f"Last audio message from Baba: {last_audio_from_baba}")
            print(f"Message ID: {last_audio_from_baba.id}")
            print(f"Timestamp: {last_audio_from_baba.timestamp}")
            print(f"Content: {last_audio_from_baba.content}")
            # download the audio file
            audio_path = download_media(last_audio_from_baba.id, chat.jid)
            print(f"Audio path: {audio_path}")
            if audio_path:
                # Convert Opus OGG to MP3 for better compatibility
                try:
                    mp3_path = convert_opus_ogg_to_mp3(audio_path)
                    print(f"Converted to MP3: {mp3_path}")
                    print(f"You can now play this MP3 file on any device: {mp3_path}")

                    # Transcribe the audio
                    try:
                        print("Transcribing audio...")
                        transcribed_text = transcribe_audio(mp3_path)
                        if transcribed_text:
                            print(f"Transcribed text: {transcribed_text}")
                        else:
                            print(
                                "No text was transcribed (audio might be silent or unclear)"
                            )
                    except Exception as e:
                        print(f"Failed to transcribe audio: {e}")

                except Exception as e:
                    print(f"Failed to convert to MP3: {e}")
            else:
                print("Failed to download audio file")
            # read the mp3 file and play it
            audio_data, sample_rate = sf.read(mp3_path)
            print(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")
            # play the audio
            sd.play(audio_data, sample_rate)
            sd.wait()
        else:
            print("No audio messages found from Baba")

    #     translation = Translation(
    #         source_text=text_to_baba,
    #         source_lang="English",
    #         target_lang="Cantonese",
    #         country="Hong Kong",
    #     )
    #     translated_text = translation.translate()
    #     audio = text_to_speech(translated_text, sid=0, language="YUE")
    #     output_dir = Path("outputs")
    #     # generate a unique id for the output file
    #     output_dir = output_dir / baba_jid
    #     output_dir.mkdir(exist_ok=True)
    #     unique_id = uuid.uuid4()
    #     output_path = output_dir / f"{unique_id}.wav"
    #     if audio is not None:
    #         sf.write(output_path, audio, 44100)
    #         print(f"Audio saved as {output_path}")
    #         # Optionally play the audio
    #         sd.play(audio, samplerate=44100)
    #         sd.wait()
    #     else:
    #         print("No audio to play or save.")
    #     # send the audio file to Baba
    #     success, status_message = send_audio_message(baba_jid, str(output_path))
    #     if success:
    #         print(f"Audio message sent successfully: {status_message}")
    #     else:
    #         print(f"Failed to send audio message: {status_message}")

    else:
        print("Baba's JID not found.")
