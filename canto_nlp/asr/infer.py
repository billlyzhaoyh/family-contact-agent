# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition", model="alvanlii/whisper-small-cantonese"
)
