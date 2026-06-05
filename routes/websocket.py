"""WebSocket route for real-time chat with LLM."""


import logging
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from models.schemas import UserMessage, AgentResponse, ErrorResponse
from services.session_manager import session_manager, serialize_session
from services.llm_service import llm_service
from serializers import json_dumps

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for chat.

    Args:
        websocket: WebSocket connection
        session_id: Unique identifier for the session

    Protocol:
        Client sends: {"text": "user message"}
        Server responds: {"response": "ai response", "session_id": "...", "message_count": N}
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for session: {session_id}")

    # Create or retrieve session
    session = session_manager.create_session(session_id)
    logger.debug(f"Session created: {serialize_session(session)}")

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received data from {session_id}: {data[:100]}")
            try:
                user_message = UserMessage(**json.loads(data))
                session_manager.add_message(session_id, "user", user_message.text)
                logger.info(f"User message added to session {session_id}")
                logger.debug(f"Session after user message: {serialize_session(session_manager.get_session(session_id))}")
                history = session_manager.get_conversation_history(session_id)
                logger.debug(f"Requesting streaming LLM response for session {session_id}")

                # Stream response from LLM
                response_chunks = []
                async for chunk in llm_service.generate_response_stream(
                    user_input=user_message.text,
                    conversation_history=history[:-1],
                ):
                    response_chunks.append(chunk)
                    partial_response = ''.join(response_chunks)
                    updated_session = session_manager.get_session(session_id)
                    agent_response = AgentResponse(
                        response=partial_response,
                        session_id=session_id,
                        message_count=updated_session.message_count if updated_session else 0,
                    )
                    response_payload = {
                        **agent_response.model_dump(),
                        "session": serialize_session(updated_session) if updated_session else None,
                        "stream": True
                    }
                    await websocket.send_text(json_dumps(response_payload))

                # Add assistant response to history (full response)
                full_response = ''.join(response_chunks)
                session_manager.add_message(session_id, "assistant", full_response)
                logger.debug(f"Session after assistant message: {serialize_session(session_manager.get_session(session_id))}")

            except json.JSONDecodeError:
                error = ErrorResponse(
                    error="Invalid JSON format. Expected: {\"text\": \"your message\"}",
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
