import settings

logger = settings.get_logger(__name__)



class TTSService:
    """Lazily loads Qwen TTS model and synthesizes mono16-bit PCM (WAV is optional packaging)."""

    def __init__(self):
        self.enabled = settings.TTS_ENABLED
        self._model = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self):
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is not None:
                return self._model

            if not self.enabled:
                raise RuntimeError("TTS is disabled. Set TTS_ENABLED=true to enable.")

            def _load():
                import torch
                from qwen_tts import Qwen3TTSModel

                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                dtype = dtype_map.get(settings.TTS_DTYPE.lower(), torch.bfloat16)

                logger.info("Loading Qwen TTS model: %s", settings.TTS_MODEL_NAME)
                preferred_attn = settings.TTS_ATTN_IMPL
                try:
                    return Qwen3TTSModel.from_pretrained(
                        settings.TTS_MODEL_NAME,
                        device_map=settings.TTS_DEVICE_MAP,
                        dtype=dtype,
                    )
                except Exception as exc:
                    err = str(exc).lower()
                    flash_requested = preferred_attn == "flash_attention_2"
                    flash_missing = "flash_attn" in err or "flashattention" in err
                    if flash_requested and flash_missing:
                        logger.warning(
                            "FlashAttention2 unavailable; falling back to eager attention."
                        )
                        return Qwen3TTSModel.from_pretrained(
                            settings.TTS_MODEL_NAME,
                            device_map=settings.TTS_DEVICE_MAP,
                            dtype=dtype,
                            attn_implementation="eager",
                        )
                    raise

            self._model = await asyncio.to_thread(_load)
            logger.info("Qwen TTS model loaded")
            return self._model


    @staticmethod
    def _waveform_to_pcm16_mono(waveform) -> bytes:
        """qwen-tts float waveform [-1, 1] → mono int16 little-endian bytes."""
        clipped = waveform.clip(-1.0, 1.0)
        pcm = (clipped * 32767).astype("int16")
        return pcm.tobytes()

    @staticmethod
    def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
        """Wrap mono s16le PCM in a WAV container (for file download or wav codec clients)."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    async def synthesize_pcm(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        """Generate speech; return mono int16 PCM bytes and sample rate (no WAV container)."""
        model = await self._ensure_model()
        tts_speaker = speaker or settings.TTS_SPEAKER
        tts_language = language or settings.TTS_LANGUAGE
        tts_instruct = instruct if instruct is not None else settings.TTS_INSTRUCT

        def _generate():
            gen_kwargs = {}
            if settings.TTS_GEN_MAX_NEW_TOKENS is not None:
                gen_kwargs["max_new_tokens"] = settings.TTS_GEN_MAX_NEW_TOKENS
            logger.info("TTS generate_custom_voice start (text_len=%s)", len(text))
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=tts_language,
                speaker=tts_speaker,
                instruct=tts_instruct,
                **gen_kwargs,
            )

            logger.info("TTS generate_custom_voice done (sample_rate=%s)", sr)
            pcm = self._waveform_to_pcm16_mono(wavs[0])
            return pcm, sr

        return await asyncio.to_thread(_generate)

    async def synthesize_wav(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        """Generate speech and return WAV file bytes + sample rate (PCM + container)."""
        pcm, sr = await self.synthesize_pcm(
            text, speaker=speaker, language=language, instruct=instruct
        )
        return self.pcm_to_wav_bytes(pcm, sr), sr

    async def check_connection(self) -> bool:
        """Basic readiness check for TTS model loading."""
        if not self.enabled:
            return False
        try:
            await self._ensure_model()
            return True
        except Exception as exc:
            logger.warning("TTS connection check failed: %s", str(exc))
            return False


tts_service = TTSService()
