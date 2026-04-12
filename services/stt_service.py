import asyncio
import io
import numpy as np
from faster_whisper import WhisperModel
import settings

logger = settings.get_logger(__name__)

class STTService:
    """Speech-to-Text service using Faster-Whisper."""

    def __init__(self):
        self.enabled = settings.STT_ENABLED
        self._model = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self):
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is not None:
                return self._model

            if not self.enabled:
                raise RuntimeError("STT is disabled. Set STT_ENABLED=true to enable.")

            def _load():
                logger.info("Loading STT model: %s (device=%s, compute_type=%s)", 
                            settings.STT_MODEL_NAME, settings.STT_DEVICE, settings.STT_COMPUTE_TYPE)
                return WhisperModel(
                    settings.STT_MODEL_NAME,
                    device=settings.STT_DEVICE,
                    compute_type=settings.STT_COMPUTE_TYPE,
                )

            self._model = await asyncio.to_thread(_load)
            logger.info("STT model loaded")
            return self._model

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe PCM audio data to text.
        
        Args:
            audio_data: Raw PCM audio (16-bit mono, 16kHz)
            
        Returns:
            Transcribed text.
        """
        model = await self._ensure_model()

        def _do_transcribe():
            # Convert PCM to float32 numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            segments, info = model.transcribe(audio_np, beam_size=5)
            text = "".join(segment.text for segment in segments).strip()
            return text

        return await asyncio.to_thread(_do_transcribe)

stt_service = STTService()
