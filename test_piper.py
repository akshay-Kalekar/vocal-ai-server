"""Smoke-test Piper TTS using the same settings as the backend (requires .env + Piper binary)."""

import asyncio

import settings
from services.tts_service import tts_service


async def main() -> None:
    if not settings.TTS_ENABLED:
        print("Set TTS_ENABLED=true and PIPER_MODEL_PATH in .env first.")
        return
    ok, err = await tts_service.verify_ready()
    print("verify_ready:", ok, err or "(ok)")
    if not ok:
        return
    pcm, sr = await tts_service.synthesize_pcm("Hello from Piper TTS.")
    print("PCM bytes:", len(pcm), "sample_rate:", sr)
    wav, _ = await tts_service.synthesize_wav("Second line test.")
    out = "piper_smoke.wav"
    with open(out, "wb") as f:
        f.write(wav)
    print("Wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
