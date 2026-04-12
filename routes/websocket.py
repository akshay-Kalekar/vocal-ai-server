"""WebSocket route for real-time chat with LLM."""


import asyncio
import logging
import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from models.schemas import UserMessage, AgentResponse, ErrorResponse
from services.session_manager import session_manager, serialize_session
from services.llm_service import llm_service
from services.tts_service import tts_service
from services.opus_encoder import (
    OPUS_SAMPLE_RATE,
    FRAME_SAMPLES,
    encode_pcm_to_opus_packets,
    opus_codec_available,
)
from serializers import json_dumps
from services.stt_service import stt_service
import pysbd
from services.vad_service import vad_service

import settings



logger = logging.getLogger(__name__)

router = APIRouter()


async def _send_safely(websocket: WebSocket, data: str):
    """Send text over websocket, catching disconnection errors."""
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(data)
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: Cannot call "send" once a close message has been sent.
        pass
    except Exception as e:
        logger.warning(f"Failed to send message safely: {e}")


def _effective_tts_codec() -> str:
    """Return 'opus' or 'wav' for outbound TTS chunks."""
    if settings.TTS_AUDIO_CODEC != "opus":

        return "wav"
    if opus_codec_available():
        return "opus"
    logger.warning(
        "TTS_AUDIO_CODEC=opus but Opus encoder unavailable "
        "(install opuslib-next and system libopus); falling back to wav"
    )
    return "wav"


async def _stream_tts_wav(
    websocket: WebSocket,
    session_id: str,
    text: str,
    *,
    context: str,
    timeout_seconds: int,
) -> None:
    """Synthesize text to WAV and send audio_start / audio_chunk / audio_end."""
    text = (text or "").strip()
    if not text:
        return

    # 1. Send audio_start
    await _send_safely(websocket, json_dumps({
        "type": "audio_start",
        "session_id": session_id,
        "context": context,
        "status": "generating",
        "format": "wav",
        "sample_rate": None,
        "text": text[:500] if context == "welcome" else text[:300],
    }))

    sample_rate = 24000 # Default fallback
    try:
        # 2. Synthesize
        pcm, sample_rate = await asyncio.wait_for(
            tts_service.synthesize_pcm(text),
            timeout=timeout_seconds,
        )
        wav_bytes = tts_service.pcm_to_wav_bytes(pcm, sample_rate)

        # 3. Stream chunks
        chunk_size = max(settings.TTS_CHUNK_SIZE, 1024)
        total_chunks = (len(wav_bytes) + chunk_size - 1) // chunk_size

        for idx in range(total_chunks):
            start = idx * chunk_size
            end = start + chunk_size
            payload = {
                "type": "audio_chunk",
                "session_id": session_id,
                "context": context,
                "format": "wav",
                "sample_rate": sample_rate,
                "sequence": idx,
                "total_chunks": total_chunks,
                "data": base64.b64encode(wav_bytes[start:end]).decode("ascii"),
                "stream": True,
            }
            await _send_safely(websocket, json_dumps(payload))

    except asyncio.TimeoutError:
        logger.warning("TTS timeout session=%s context=%s", session_id, context)
        await _send_safely(websocket, json_dumps({
            "type": "audio_error",
            "session_id": session_id,
            "context": context,
            "error": "TTS generation timed out",
        }))
    except Exception as e:
        logger.error("TTS failed session=%s context=%s: %s", session_id, context, str(e))
        await _send_safely(websocket, json_dumps({
            "type": "audio_error",
            "session_id": session_id,
            "context": context,
            "error": str(e),
        }))
    finally:
        # 4. ALWAYS send audio_end or UI gets stuck in "processing"
        await _send_safely(websocket, json_dumps({
            "type": "audio_end",
            "session_id": session_id,
            "context": context,
            "format": "wav",
            "sample_rate": sample_rate,
            "stream": False,
        }))


async def _stream_tts_opus(
    websocket: WebSocket,
    session_id: str,
    text: str,
    *,
    context: str,
    timeout_seconds: int,
) -> None:
    """Synthesize text to Opus and send audio_start / audio_chunk / audio_end."""
    text = (text or "").strip()
    if not text:
        return

    # 1. Send audio_start
    await _send_safely(websocket, json_dumps({
        "type": "audio_start",
        "session_id": session_id,
        "context": context,
        "status": "generating",
        "format": "opus",
        "sample_rate": OPUS_SAMPLE_RATE,
        "opus_frame_samples": FRAME_SAMPLES,
        "text": text[:500] if context == "welcome" else text[:300],
    }))

    try:
        # 2. Synthesize
        pcm, sr = await asyncio.wait_for(
            tts_service.synthesize_pcm(text),
            timeout=timeout_seconds,
        )
        
        # 3. Encode to Opus
        packets = await asyncio.to_thread(encode_pcm_to_opus_packets, pcm, sr)

        # 4. Stream chunks
        total_chunks = len(packets)
        for idx, pkt in enumerate(packets):
            payload = {
                "type": "audio_chunk",
                "session_id": session_id,
                "context": context,
                "format": "opus",
                "sample_rate": OPUS_SAMPLE_RATE,
                "opus_frame_samples": FRAME_SAMPLES,
                "sequence": idx,
                "total_chunks": total_chunks,
                "data": base64.b64encode(pkt).decode("ascii"),
                "stream": True,
            }
            await _send_safely(websocket, json_dumps(payload))

    except asyncio.TimeoutError:
        logger.warning("TTS timeout session=%s context=%s", session_id, context)
        await _send_safely(websocket, json_dumps({
            "type": "audio_error",
            "session_id": session_id,
            "context": context,
            "error": "TTS generation timed out",
        }))
    except Exception as e:
        logger.error("TTS failed session=%s context=%s: %s", session_id, context, str(e))
        await _send_safely(websocket, json_dumps({
            "type": "audio_error",
            "session_id": session_id,
            "context": context,
            "error": str(e),
        }))
    finally:
        # 5. ALWAYS send audio_end
        await _send_safely(websocket, json_dumps({
            "type": "audio_end",
            "session_id": session_id,
            "context": context,
            "format": "opus",
            "sample_rate": OPUS_SAMPLE_RATE,
            "opus_frame_samples": FRAME_SAMPLES,
            "stream": False,
        }))


async def _stream_tts_audio(
    websocket: WebSocket,
    session_id: str,
    text: str,
    *,
    context: str,
    timeout_seconds: int,
    codec: str,
) -> None:
    if codec == "opus":
        await _stream_tts_opus(
            websocket, session_id, text,
            context=context, timeout_seconds=timeout_seconds
        )
    else:
        await _stream_tts_wav(
            websocket, session_id, text,
            context=context, timeout_seconds=timeout_seconds
        )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time voice chat (VAD + STT + LLM + TTS)."""
    
    async def run_welcome(codec):
        if settings.TTS_ENABLED:
            try:
                await _stream_tts_audio(
                    websocket, session_id, settings.TTS_WELCOME_TEXT,
                    context="welcome", timeout_seconds=settings.TTS_WELCOME_TIMEOUT,
                    codec=codec
                )
            except Exception as e:
                logger.warning(f"Welcome TTS failed: {e}")

    try:
        await websocket.accept()
        logger.info(f"WebSocket connection: {session_id}")

        session_manager.create_session(session_id)
        tts_codec = _effective_tts_codec() if settings.TTS_ENABLED else "wav"
        segmenter = pysbd.Segmenter(language="en", clean=False)

        await _send_safely(websocket, json_dumps({
            "type": "session_ready",
            "session_id": session_id,
            "tts_enabled": settings.TTS_ENABLED,
            "tts_codec": tts_codec,
        }))

        # Start welcome task in background
        asyncio.create_task(run_welcome(tts_codec))

        state = {
            "is_processing": False,
            "audio_buffer": bytearray(),
            "vad_state": "waiting", # waiting, speaking, silence
            "silence_start": None,
            "utterance_buffer": bytearray(),
        }

        class TTSStreamer:
            def __init__(self):
                self.buffer = ""
                self.processed_text = ""

            async def push(self, chunk: str):
                self.buffer += chunk
                sentences = segmenter.segment(self.buffer)
                if len(sentences) > 1 or (sentences and any(self.buffer.endswith(p) for p in ".!?")):
                    if not any(self.buffer.endswith(p) for p in ".!?"):
                        to_process = sentences[:-1]
                        self.buffer = sentences[-1]
                    else:
                        to_process = sentences
                        self.buffer = ""
                    
                    for sentence in to_process:
                        clean_sent = sentence.strip()
                        if clean_sent and clean_sent not in self.processed_text:
                            self.processed_text += clean_sent
                            await _stream_tts_audio(
                                websocket, session_id, clean_sent,
                                context="reply", timeout_seconds=settings.TTS_REPLY_TIMEOUT,
                                codec=tts_codec
                            )

            async def finalize(self):
                if self.buffer.strip():
                    await _stream_tts_audio(
                        websocket, session_id, self.buffer.strip(),
                        context="reply", timeout_seconds=settings.TTS_REPLY_TIMEOUT,
                        codec=tts_codec
                    )
                self.buffer = ""

        async def trigger_llm_response(user_text: str):
            state["is_processing"] = True
            try:
                session_manager.add_message(session_id, "user", user_text)
                history = session_manager.get_conversation_history(session_id)
                
                tts_streamer = TTSStreamer()
                response_chunks = []
                
                async for chunk in llm_service.generate_response_stream(
                    user_input=user_text,
                    conversation_history=history[:-1],
                ):
                    # Stop if session was closed
                    if not session_manager.get_session(session_id):
                        break

                    response_chunks.append(chunk)
                    full_text = "".join(response_chunks)
                    
                    await _send_safely(websocket, json_dumps({
                        "type": "agent_response",
                        "response": full_text,
                        "session_id": session_id,
                        "stream": True
                    }))
                
                    if settings.TTS_ENABLED:
                        await tts_streamer.push(chunk)

                full_response = "".join(response_chunks)
                session_manager.add_message(session_id, "assistant", full_response)
                
                if settings.TTS_ENABLED:
                    await tts_streamer.finalize()

                await _send_safely(websocket, json_dumps({
                    "type": "agent_response",
                    "response": full_response,
                    "session_id": session_id,
                    "stream": False,
                    "tts_follows": False 
                }))
            except Exception as e:
                logger.error(f"LLM Error: {e}")
            finally:
                state["is_processing"] = False

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    await _send_safely(websocket, json_dumps({"type": "pong"}))
                    continue
                
                user_text = data.get("text", "")
                if user_text and not state["is_processing"]:
                    asyncio.create_task(trigger_llm_response(user_text))

            elif "bytes" in message:
                if state["is_processing"]:
                    continue
                
                audio_chunk = message["bytes"]
                state["audio_buffer"].extend(audio_chunk)

                frame_size = 960  # 30ms @ 16kHz
                while len(state["audio_buffer"]) >= frame_size:
                    frame = state["audio_buffer"][:frame_size]
                    del state["audio_buffer"][:frame_size]
                    
                    is_speech = vad_service.is_speech(frame, 16000)
                    current_time = asyncio.get_event_loop().time()
                    
                    if is_speech:
                        if state["vad_state"] == "waiting":
                            logger.info("Speech started")
                            state["utterance_buffer"] = bytearray()
                        state["vad_state"] = "speaking"
                        state["silence_start"] = None
                        state["utterance_buffer"].extend(frame)
                    else:
                        if state["vad_state"] == "speaking":
                            state["vad_state"] = "silence"
                            state["silence_start"] = current_time
                        
                        if state["vad_state"] == "silence":
                            state["utterance_buffer"].extend(frame)
                            if current_time - state["silence_start"] > settings.STT_SILENCE_THRESHOLD:
                                logger.info("Silence detected, triggering STT")
                                utterance = bytes(state["utterance_buffer"])
                                state["utterance_buffer"] = bytearray()
                                state["vad_state"] = "waiting"
                                
                                if settings.STT_ENABLED:
                                    try:
                                        if not session_manager.get_session(session_id):
                                            break
                                        text = await stt_service.transcribe(utterance)
                                        if text:
                                            logger.info(f"User: {text}")
                                            await _send_safely(websocket, json_dumps({
                                                "type": "user_speech",
                                                "text": text,
                                                "session_id": session_id
                                            }))
                                            asyncio.create_task(trigger_llm_response(text))
                                    except Exception as e:
                                        logger.error(f"STT Error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Disconnect: {session_id}")
    except Exception as e:
        logger.error(f"WS Error {session_id}: {e}", exc_info=True)
    finally:
        session_manager.close_session(session_id)
