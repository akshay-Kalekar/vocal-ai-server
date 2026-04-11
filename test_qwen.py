import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load the model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cpu",          # ❗ force CPU
    torch_dtype="float32",     # ❗ bfloat16 not safe on most CPUs
)

# Generate speech with specific instructions
wavs, sr = model.generate_custom_voice(
    text="Hello, how are you? My name is Akshay. Such a pleasure to meet you.",
    language="English", 
    speaker="Ryan",
    instruct="speak in a sweet voice", 
)

# Save the generated audio
sf.write("output_custom_voice.wav", wavs[0], sr)
