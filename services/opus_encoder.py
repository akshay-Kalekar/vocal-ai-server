"""Encode TTS WAV output to Opus packets (48 kHz, 20 ms frames) for WebSocket streaming."""

from __future__ import annotations

import audioop
import io
import wave
from typing import Iterator, List, Tuple

# Opus uses 48 kHz internally; we resample PCM to this rate before encoding.
OPUS_SAMPLE_RATE = 48000
# 20 ms @ 48 kHz mono int16 — matches common Opus frame size.
FRAME_SAMPLES = 960
FRAME_BYTES = FRAME_SAMPLES * 2

_codec_ok: bool | None = None


def opus_codec_available() -> bool:
    """True if libopus is loadable and encoder can be created."""
    global _codec_ok
    if _codec_ok is not None:
        return _codec_ok
    try:
        import opuslib_next as opuslib

        opuslib.Encoder(OPUS_SAMPLE_RATE, 1, opuslib.APPLICATION_AUDIO)
        _codec_ok = True
    except Exception:
        _codec_ok = False
    return _codec_ok


def _pcm_resample_mono(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm
    state = None
    out: List[bytes] = []
    chunk_size = 32768
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        chunk, state = audioop.ratecv(chunk, 2, 1, src_rate, dst_rate, state)
        out.append(chunk)
    if state is not None:
        tail, _ = audioop.ratecv(b"", 2, 1, src_rate, dst_rate, state)
        if tail:
            out.append(tail)
    return b"".join(out)


def wav_to_pcm_mono_int16(wav_bytes: bytes) -> Tuple[bytes, int]:
    """Read WAV bytes; return mono s16le PCM and original sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        pcm = w.readframes(w.getnframes())
    if nch == 2:
        pcm = audioop.tomono(pcm, sw, 0.5, 0.5)
    elif nch != 1:
        raise ValueError(f"unsupported channel count: {nch}")
    if sw != 2:
        pcm = audioop.lin2lin(pcm, sw, 2, 1)
    return pcm, sr


def iter_opus_packets_from_pcm(pcm: bytes, sample_rate: int) -> Iterator[bytes]:
    """Yield Opus packets from mono int16 PCM (any rate; resampled to 48 kHz)."""
    import opuslib_next as opuslib

    pcm_48k = _pcm_resample_mono(pcm, sample_rate, OPUS_SAMPLE_RATE)
    encoder = opuslib.Encoder(OPUS_SAMPLE_RATE, 1, opuslib.APPLICATION_AUDIO)
    offset = 0
    while offset + FRAME_BYTES <= len(pcm_48k):
        frame = pcm_48k[offset : offset + FRAME_BYTES]
        offset += FRAME_BYTES
        yield encoder.encode(frame, FRAME_SAMPLES)
    rem = len(pcm_48k) - offset
    if rem > 0:
        pad = FRAME_BYTES - rem
        frame = pcm_48k[offset:] + b"\x00" * pad
        yield encoder.encode(frame, FRAME_SAMPLES)


def iter_opus_packets_from_wav(wav_bytes: bytes) -> Iterator[bytes]:
    pcm, sr = wav_to_pcm_mono_int16(wav_bytes)
    yield from iter_opus_packets_from_pcm(pcm, sr)


def encode_pcm_to_opus_packets(pcm: bytes, sample_rate: int) -> List[bytes]:
    return list(iter_opus_packets_from_pcm(pcm, sample_rate))


def encode_wav_to_opus_packets(wav_bytes: bytes) -> List[bytes]:
    pcm, sr = wav_to_pcm_mono_int16(wav_bytes)
    return encode_pcm_to_opus_packets(pcm, sr)
