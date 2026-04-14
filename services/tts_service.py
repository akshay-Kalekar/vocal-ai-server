"""TTS via Piper (ONNX, CPU-friendly). Requires `piper` binary and a voice model."""

from __future__ import annotations

import asyncio
import json
import io
import re
import shlex
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import settings

logger = settings.get_logger(__name__)


def _voice_json_path(model_path: str) -> Path:
    """Piper pairs `voice.onnx` with `voice.onnx.json`."""
    return Path(f"{model_path}.json")


def _normalize_piper_text(text: str) -> str:
    """Piper reads stdin line-oriented; collapse newlines so the full utterance is spoken."""
    text = (text or "").strip()
    if not text:
        return text
    text = re.sub(r"[\r\n]+", " ", text)
    return text


class TTSService:
    """Runs the `piper` CLI to produce mono s16le PCM (WAV wrapping is optional)."""

    def __init__(self):
        self.enabled = settings.TTS_ENABLED
        self._gen_lock = asyncio.Lock()
        self._ready_lock = asyncio.Lock()
        self._sample_rate: Optional[int] = None
        self._extra_args: List[str] = shlex.split(settings.PIPER_EXTRA_ARGS)

    def _executable(self) -> str:
        exe = settings.PIPER_EXECUTABLE.strip()
        if not exe:
            raise RuntimeError("PIPER_EXECUTABLE is empty")
        path = Path(exe)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(exe)
        if not found:
            raise RuntimeError(
                f"Piper executable not found: {exe!r} (install Piper and/or set PIPER_EXECUTABLE)"
            )
        return found

    def _model_path(self) -> str:
        mp = (settings.PIPER_MODEL_PATH or "").strip()
        if not mp:
            raise RuntimeError(
                "PIPER_MODEL_PATH is not set (path to a Piper .onnx voice, e.g. en_US-lessac-medium.onnx)"
            )
        p = Path(mp).expanduser()
        if not p.is_file():
            raise RuntimeError(f"Piper model file not found: {p}")
        return str(p.resolve())

    def _load_sample_rate(self, model_path: str) -> int:
        json_override = (settings.PIPER_VOICE_JSON or "").strip()
        jp = Path(json_override).expanduser() if json_override else _voice_json_path(model_path)
        if not jp.is_file():
            logger.warning(
                "Piper voice JSON missing at %s; assuming sample_rate=22050. "
                "Download the matching .onnx.json next to your .onnx model.",
                jp,
            )
            return 22050
        with open(jp, encoding="utf-8") as f:
            cfg = json.load(f)
        audio = cfg.get("audio") or {}
        sr = audio.get("sample_rate") or cfg.get("sample_rate")
        if not isinstance(sr, int) or sr < 1:
            raise RuntimeError(f"Invalid or missing sample_rate in {jp}")
        return sr

    async def _ensure_ready(self) -> int:
        async with self._ready_lock:
            if self._sample_rate is not None:
                return self._sample_rate
            if not self.enabled:
                raise RuntimeError("TTS is disabled. Set TTS_ENABLED=true to enable.")
            model_path = self._model_path()
            self._executable()
            self._sample_rate = self._load_sample_rate(model_path)
            logger.info(
                "Piper TTS ready (model=%s, sample_rate=%s)",
                model_path,
                self._sample_rate,
            )
            return self._sample_rate

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

    def _synthesize_pcm_sync(self, text: str) -> Tuple[bytes, int]:
        text = _normalize_piper_text(text)
        if not text:
            return b"", self._sample_rate or 22050

        exe = self._executable()
        model = self._model_path()
        sr = self._sample_rate or self._load_sample_rate(model)

        cmd = [exe, "--model", model, "--output_raw"]
        json_override = (settings.PIPER_VOICE_JSON or "").strip()
        if json_override:
            cmd.extend(["--config", str(Path(json_override).expanduser().resolve())])
        cmd.extend(self._extra_args)
        timeout = max(5, int(settings.TTS_REPLY_TIMEOUT))
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Piper subprocess exceeded {timeout}s") from e

        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or f"Piper exited with code {proc.returncode}")

        pcm = proc.stdout
        if not pcm:
            raise RuntimeError("Piper produced no audio output")
        return pcm, sr

    async def synthesize_pcm(self, text: str) -> Tuple[bytes, int]:
        """Generate speech; return mono int16 PCM bytes and sample rate (no WAV container)."""
        await self._ensure_ready()
        async with self._gen_lock:
            return await asyncio.to_thread(self._synthesize_pcm_sync, text)

    async def synthesize_wav(self, text: str) -> Tuple[bytes, int]:
        """Generate speech and return WAV file bytes + sample rate."""
        pcm, sr = await self.synthesize_pcm(text)
        return self.pcm_to_wav_bytes(pcm, sr), sr

    async def verify_ready(self) -> tuple[bool, str]:
        """Return (True, empty string) if Piper can run, else (False, reason)."""
        if not self.enabled:
            return True, ""
        try:
            await self._ensure_ready()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    async def check_connection(self) -> bool:
        """Verify Piper binary, model, and voice JSON / sample rate."""
        ok, _ = await self.verify_ready()
        return ok


tts_service = TTSService()
