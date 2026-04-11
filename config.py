import os
from dotenv import load_dotenv

load_dotenv()

# Ollama/Local LLM Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")

# Session Management
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour in seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Qwen TTS Configuration
TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() == "true"
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "Ryan")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "English")
TTS_INSTRUCT = os.getenv("TTS_INSTRUCT", "")
TTS_WELCOME_TEXT = os.getenv(
    "TTS_WELCOME_TEXT",
    "Welcome. You are connected to the low cost voice agent.",
)
TTS_DEVICE_MAP = os.getenv("TTS_DEVICE_MAP", "cpu")
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32")
TTS_ATTN_IMPL = os.getenv("TTS_ATTN_IMPL", "flash_attention_2")
TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "8192"))
# Max seconds to wait for welcome TTS before sending audio_error and continuing chat.
TTS_WELCOME_TIMEOUT = int(os.getenv("TTS_WELCOME_TIMEOUT", "600"))
# Max seconds for synthesizing assistant reply audio (defaults to welcome timeout).
TTS_REPLY_TIMEOUT = int(os.getenv("TTS_REPLY_TIMEOUT", str(TTS_WELCOME_TIMEOUT)))
# Truncate assistant text before TTS to avoid very long synthesis.
TTS_REPLY_MAX_CHARS = int(os.getenv("TTS_REPLY_MAX_CHARS", "2000"))
# Optional cap passed to generate_custom_voice (smaller = faster, shorter audio).
_tok = os.getenv("TTS_GEN_MAX_NEW_TOKENS", "").strip()
TTS_GEN_MAX_NEW_TOKENS = int(_tok) if _tok else None
# WebSocket TTS payload: "opus" (libopus + opuslib-next) or "wav" (raw WAV chunks).
TTS_AUDIO_CODEC = os.getenv("TTS_AUDIO_CODEC", "opus").strip().lower()

# API Configuration
API_TITLE = "Low Cost Voice Agent"
API_VERSION = "0.1.0"
