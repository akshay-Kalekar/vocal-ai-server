"""WebSocket test client: text stream + TTS audio (welcome + per-reply)."""

import asyncio
import base64
import json
import os
import wave
import websockets
import sys
from pathlib import Path

try:
    import opuslib_next as opuslib
except ImportError:
    opuslib = None

WELCOME_RECV_TIMEOUT = float(os.getenv("WELCOME_RECV_TIMEOUT", "650"))
REPLY_AUDIO_TIMEOUT = float(os.getenv("REPLY_AUDIO_TIMEOUT", "650"))

OPUS_DECODE_RATE = 48000
OPUS_FRAME_SAMPLES = 960


def _opus_packets_to_pcm(packets: list[bytes]) -> bytes:
    if not opuslib:
        raise RuntimeError(
            "Opus chunks require opuslib-next (pip install opuslib-next) and system libopus"
        )
    decoder = opuslib.Decoder(OPUS_DECODE_RATE, 1)
    return b"".join(decoder.decode(pkt, OPUS_FRAME_SAMPLES) for pkt in packets)


def _write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


async def receive_audio_stream(websocket, prefix: str, session_id: str, turn: int | None = None):
    """Receive audio_start → chunks → audio_end (or audio_error). Saves WAV."""
    chunks = {}
    sample_rate = None
    audio_format = "wav"
    suffix = f"_{turn}" if turn is not None else ""
    out_name = f"{prefix}{suffix}_{session_id}.wav"

    while True:
        raw = await websocket.recv()
        data = json.loads(raw)

        if data.get("type") == "audio_start":
            ctx = data.get("context", "")
            audio_format = data.get("format") or "wav"
            print(
                f"  🔊 Audio generating ({ctx}, format={audio_format})…",
                flush=True,
            )
            continue

        if data.get("type") == "audio_chunk":
            idx = data.get("sequence", 0)
            chunks[idx] = base64.b64decode(data["data"])
            sample_rate = data.get("sample_rate")
            audio_format = data.get("format") or audio_format
            continue

        if data.get("type") == "audio_end":
            if chunks:
                out_path = Path(out_name)
                ordered = [chunks[i] for i in sorted(chunks.keys())]
                if audio_format == "opus":
                    try:
                        pcm = _opus_packets_to_pcm(ordered)
                        sr = sample_rate or OPUS_DECODE_RATE
                        _write_pcm_wav(out_path, pcm, sr)
                        print(
                            f"✓ Opus decoded to WAV ({len(pcm)} bytes PCM, sr={sr}): "
                            f"{out_path.resolve()}\n",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"✗ Opus decode failed: {e}\n", flush=True)
                else:
                    wav_bytes = b"".join(ordered)
                    out_path.write_bytes(wav_bytes)
                    print(
                        f"✓ Audio saved ({len(wav_bytes)} bytes, sr={sample_rate}): {out_path.resolve()}\n",
                        flush=True,
                    )
            else:
                print("ℹ audio_end with no chunks\n", flush=True)
            return

        if data.get("type") == "audio_error":
            print(f"✗ Audio error [{data.get('context')}]: {data.get('error')}\n", flush=True)
            return

        # Protocol violation for this client: unexpected frame during audio
        print(f"⚠ Unexpected frame during audio wait: {list(data.keys())}\n", flush=True)
        return


async def test_websocket(session_id: str = "test-session-1"):
    uri = f"ws://localhost:8000/ws/{session_id}"

    print(f"Connecting to {uri}")
    print("Type 'exit' to quit\n")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected\n")

            raw0 = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            hello = json.loads(raw0)
            if hello.get("type") == "session_ready":
                codec = hello.get("tts_codec")
                if codec:
                    print(f"ℹ TTS codec: {codec}\n", flush=True)
                if hello.get("tts_enabled"):
                    try:
                        await asyncio.wait_for(
                            receive_audio_stream(websocket, "welcome", session_id),
                            timeout=WELCOME_RECV_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        print("ℹ Welcome audio timed out (very slow TTS)\n")
                else:
                    print("ℹ TTS disabled — no welcome audio\n")
            else:
                # Older server without session_ready: treat as start of audio or ignore
                print(f"⚠ Expected session_ready, got keys={list(hello.keys())}\n")

            turn = 0
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    print("Closing connection...")
                    break
                if not user_input:
                    continue

                await websocket.send(json.dumps({"text": user_input, "session_id": session_id}))
                print("→ Sent\n")

                partial = ""
                message_count = None

                while True:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(raw)

                    if data.get("type") == "audio_error":
                        print(f"✗ {data.get('error')}\n")
                        break

                    if data.get("error") and "response" not in data:
                        print(f"✗ Error: {data.get('error')}\n")
                        break

                    if data.get("stream") is True and "response" in data:
                        new_text = data["response"][len(partial) :]
                        print(new_text, end="", flush=True)
                        partial = data["response"]
                        message_count = data.get("message_count")
                        continue

                    if data.get("stream") is False and "response" in data:
                        if partial:
                            print(f"\n(Text done, message_count={message_count})\n")
                        turn += 1
                        if data.get("tts_follows"):
                            try:
                                await asyncio.wait_for(
                                    receive_audio_stream(
                                        websocket, "reply", session_id, turn=turn
                                    ),
                                    timeout=REPLY_AUDIO_TIMEOUT,
                                )
                            except asyncio.TimeoutError:
                                print("ℹ Reply audio timed out\n")
                        break

                    print(f"⚠ Unexpected: {data}\n")
                    break

    except ConnectionRefusedError:
        print("✗ Server not running? (fastapi dev)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "test-session-1"
    asyncio.run(test_websocket(sid))
