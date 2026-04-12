"""WebSocket route for real-time chat with LLM."""


import asyncio
import logging
import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
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
import settings


logger = logging.getLogger(__name__)

router = APIRouter()


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
    """Synthesize text to WAV and send audio_start / audio_chunk / audio_end (or audio_error)."""
    text = (text or "").strip()
    if not text:
        return

    max_len = getattr(settings, "TTS_REPLY_MAX_CHARS", 2000)

    if context == "reply" and len(text) > max_len:
        text = text[:max_len]

    await websocket.send_text(
        json_dumps(
            {
                "type": "audio_start",
                "session_id": session_id,
                "context": context,
                "status": "generating",
                "format": "wav",
                "sample_rate": None,
                "text": text[:500] if context == "welcome" else text[:300],
            }
        )
    )
    try:
        pcm, sample_rate = await asyncio.wait_for(
            tts_service.synthesize_pcm(text),
            timeout=timeout_seconds,
        )
        wav_bytes = tts_service.pcm_to_wav_bytes(pcm, sample_rate)
    except asyncio.TimeoutError:
        err = (
            f"TTS ({context}) exceeded {timeout_seconds}s "
            "(try shorter text, TTS_REPLY_MAX_CHARS, or TTS_GEN_MAX_NEW_TOKENS)"
        )
        logger.warning("TTS timeout session=%s context=%s", session_id, context)
        await websocket.send_text(
            json_dumps(
                {
                    "type": "audio_error",
                    "session_id": session_id,
                    "context": context,
                    "error": err,
                }
            )
        )
        return
    except Exception as e:
        logger.warning("TTS failed session=%s context=%s: %s", session_id, context, str(e))
        await websocket.send_text(
            json_dumps(
                {
                    "type": "audio_error",
                    "session_id": session_id,
                    "context": context,
                    "error": str(e),
                }
            )
        )
        return

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
        await websocket.send_text(json_dumps(payload))

    await websocket.send_text(
        json_dumps(
            {
                "type": "audio_end",
                "session_id": session_id,
                "context": context,
                "format": "wav",
                "sample_rate": sample_rate,
                "stream": False,
            }
        )
    )


async def _stream_tts_opus(
    websocket: WebSocket,
    session_id: str,
    text: str,
    *,
    context: str,
    timeout_seconds: int,
) -> None:
    """Synthesize to PCM, encode to Opus packets, stream over WebSocket."""
    text = (text or "").strip()
    if not text:
        return

    max_len = getattr(settings, "TTS_REPLY_MAX_CHARS", 2000)

    if context == "reply" and len(text) > max_len:
        text = text[:max_len]

    await websocket.send_text(
        json_dumps(
            {
                "type": "audio_start",
                "session_id": session_id,
                "context": context,
                "status": "generating",
                "format": "opus",
                "sample_rate": OPUS_SAMPLE_RATE,
                "opus_frame_samples": FRAME_SAMPLES,
                "text": text[:500] if context == "welcome" else text[:300],
            }
        )
    )
    try:
        pcm, sr = await asyncio.wait_for(
            tts_service.synthesize_pcm(text),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        err = (
            f"TTS ({context}) exceeded {timeout_seconds}s "
            "(try shorter text, TTS_REPLY_MAX_CHARS, or TTS_GEN_MAX_NEW_TOKENS)"
        )
        logger.warning("TTS timeout session=%s context=%s", session_id, context)
        await websocket.send_text(
            json_dumps(
                {
                    "type": "audio_error",
                    "session_id": session_id,
                    "context": context,
                    "error": err,
                }
            )
        )
        return
    except Exception as e:
        logger.warning("TTS failed session=%s context=%s: %s", session_id, context, str(e))
        await websocket.send_text(
            json_dumps(
                {
                    "type": "audio_error",
                    "session_id": session_id,
                    "context": context,
                    "error": str(e),
                }
            )
        )
        return

    try:
        packets = await asyncio.to_thread(encode_pcm_to_opus_packets, pcm, sr)
    except Exception as e:
        logger.warning("Opus encode failed session=%s context=%s: %s", session_id, context, str(e))
        await websocket.send_text(
            json_dumps(
                {
                    "type": "audio_error",
                    "session_id": session_id,
                    "context": context,
                    "error": str(e),
                }
            )
        )
        return

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
        await websocket.send_text(json_dumps(payload))

    await websocket.send_text(
        json_dumps(
            {
                "type": "audio_end",
                "session_id": session_id,
                "context": context,
                "format": "opus",
                "sample_rate": OPUS_SAMPLE_RATE,
                "opus_frame_samples": FRAME_SAMPLES,
                "stream": False,
            }
        )
    )


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
            websocket,
            session_id,
            text,
            context=context,
            timeout_seconds=timeout_seconds,
        )
    else:
        await _stream_tts_wav(
            websocket,
            session_id,
            text,
            context=context,
            timeout_seconds=timeout_seconds,
        )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for chat.

    Args:
        websocket: WebSocket connection
        session_id: Unique identifier for the session

    Protocol:
        Client sends: {"text": "user message"}
        Server responds: JSON text stream (stream true/false), then if TTS_ENABLED:
        audio_start / audio_chunk / audio_end with context "welcome" (on connect)
        or context "reply" (after each assistant message).
        Audio chunks use format "opus" (preferred) or "wav" per session_ready.tts_codec.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")

    # Create or retrieve session
    session_manager.create_session(session_id)

    tts_codec = _effective_tts_codec() if settings.TTS_ENABLED else "wav"


    await websocket.send_text(
        json_dumps(
            {
                "type": "session_ready",
                "session_id": session_id,
                "tts_enabled": settings.TTS_ENABLED,

                "tts_codec": tts_codec,
            }
        )
    )

    if settings.TTS_ENABLED:

        try:
            await _stream_tts_audio(
                websocket,
                session_id,
                settings.TTS_WELCOME_TEXT,

                context="welcome",
                timeout_seconds=settings.TTS_WELCOME_TIMEOUT,

                codec=tts_codec,
            )
        except Exception as e:
            logger.warning("Failed welcome TTS for %s: %s", session_id, str(e))
            await websocket.send_text(
                json_dumps(
                    {
                        "type": "audio_error",
                        "session_id": session_id,
                        "context": "welcome",
                        "error": str(e),
                    }
                )
            )

    try:
        while True:
            data = await websocket.receive_text()
            try:
                user_message = UserMessage(**json.loads(data))
                session_manager.add_message(session_id, "user", user_message.text)
                history = session_manager.get_conversation_history(session_id)

                # Stream response from LLM
                response_chunks = []
                async for chunk in llm_service.generate_response_stream(
                    user_input=user_message.text,
                    conversation_history=history[:-1],
                ):
                    response_chunks.append(chunk)
                    partial_response = "".join(response_chunks)
                    updated_session = session_manager.get_session(session_id)
                    agent_response = AgentResponse(
                        response=partial_response,
                        session_id=session_id,
                        message_count=updated_session.message_count if updated_session else 0,
                    )
                    response_payload = {
                        **agent_response.model_dump(),
                        "session": serialize_session(updated_session) if updated_session else None,
                        "stream": True,
                    }
                    await websocket.send_text(json_dumps(response_payload))

                full_response = "".join(response_chunks)
                session_manager.add_message(session_id, "assistant", full_response)
                updated_session = session_manager.get_session(session_id)

                final_response = AgentResponse(
                    response=full_response,
                    session_id=session_id,
                    message_count=updated_session.message_count if updated_session else 0,
                )
                will_tts = bool(
                    settings.TTS_ENABLED and full_response.strip()

                )
                final_payload = {
                    **final_response.model_dump(),
                    "session": serialize_session(updated_session) if updated_session else None,
                    "stream": False,
                    "tts_follows": will_tts,
                }
                await websocket.send_text(json_dumps(final_payload))

                if will_tts:
                    await _stream_tts_audio(
                        websocket,
                        session_id,
                        full_response,
                        context="reply",
                        timeout_seconds=settings.TTS_REPLY_TIMEOUT,

                        codec=tts_codec,
                    )

            except json.JSONDecodeError:
                error = ErrorResponse(
                    error='Invalid JSON format. Expected: {"text": "your message"}',
                    session_id=session_id,
                )
                await websocket.send_text(json_dumps(error.model_dump()))
                logger.warning(f"Invalid JSON received from {session_id}")

            except ValueError as e:
                error = ErrorResponse(
                    error=f"Invalid message format: {str(e)}",
                    session_id=session_id,
                )
                await websocket.send_text(json_dumps(error.model_dump()))
                logger.warning(f"Validation error for {session_id}: {str(e)}")

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(f"Session {session_id} error: {error_msg}")
                error = ErrorResponse(
                    error=error_msg,
                    session_id=session_id,
                )
                await websocket.send_text(json_dumps(error.model_dump()))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        session_manager.close_session(session_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for session {session_id}: {str(e)}")
        session_manager.close_session(session_id)
