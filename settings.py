import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Ollama/Local LLM Configuration ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2")

# --- Session Management ---
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour in seconds

# --- Logging Configuration ---
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR.upper(), logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Keep dependency logs concise unless explicitly debugging.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name: str):
    """Utility to get a logger with the project's configuration."""
    return logging.getLogger(name)

logger = get_logger(__name__)

# --- Piper TTS (CPU-friendly; requires `piper` binary + .onnx voice) ---
# Voices: https://github.com/rhasspy/piper/releases (download .onnx + matching .onnx.json)
TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() == "true"
PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE", "piper")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "").strip()
# Optional: explicit path to voice JSON (default: {PIPER_MODEL_PATH}.json)
PIPER_VOICE_JSON = os.getenv("PIPER_VOICE_JSON", "").strip()
# Extra CLI args, e.g. '--noise-scale 0.667 --length-scale 1.0'
PIPER_EXTRA_ARGS = os.getenv("PIPER_EXTRA_ARGS", "")
TTS_WELCOME_TEXT = os.getenv(
    "TTS_WELCOME_TEXT",
    "Welcome. You are connected to the low cost voice agent.",
)
TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "8192"))
TTS_WELCOME_TIMEOUT = int(os.getenv("TTS_WELCOME_TIMEOUT", "600"))
TTS_REPLY_TIMEOUT = int(os.getenv("TTS_REPLY_TIMEOUT", str(TTS_WELCOME_TIMEOUT)))
TTS_REPLY_MAX_CHARS = int(os.getenv("TTS_REPLY_MAX_CHARS", "2000"))
TTS_AUDIO_CODEC = os.getenv("TTS_AUDIO_CODEC", "opus").strip().lower()

# --- STT Configuration ---
STT_ENABLED = os.getenv("STT_ENABLED", "true").lower() == "true"
STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "default")
STT_SILENCE_THRESHOLD = float(os.getenv("STT_SILENCE_THRESHOLD", "0.8"))


# --- API Configuration ---
API_TITLE = "Low Cost Voice Agent"
API_VERSION = "0.1.0"
API_DESCRIPTION = "Low Cost Voice Agent - Real-time chat with local LLM"
