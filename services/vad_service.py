import webrtcvad
import settings

logger = settings.get_logger(__name__)

class VADService:
    """Voice Activity Detection service using webrtcvad."""

    def __init__(self, aggressiveness: int = 3):
        """Initialize VAD.
        
        Args:
            aggressiveness: 0 (least aggressive) to 3 (most aggressive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        # webrtcvad supports 8000, 16000, 32000, 48000 Hz
        # and frames of 10, 20, or 30 ms.
        self.sample_rate = 16000 
        self.frame_duration_ms = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 2 bytes per sample

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Check if a single chunk contains speech.
        
        Args:
            audio_chunk: Raw PCM audio data (16-bit mono)
            sample_rate: Sample rate of the audio
            
        Returns:
            True if speech is detected, False otherwise.
        """
        try:
            return self.vad.is_speech(audio_chunk, sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False

    def process_stream(self, session_id: str, audio_data: bytes):
        """Process incoming stream for a session and detect voice end.
        (Higher level logic can be added here)
        """
        pass

# Global VAD service instance
vad_service = VADService()
