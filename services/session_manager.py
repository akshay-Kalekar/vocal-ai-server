import settings
from typing import Dict, Optional
from datetime import datetime
from models.schemas import Message, SessionData

logger = settings.get_logger(__name__)



def serialize_session(session: SessionData):
    """Serialize a SessionData object to a dictionary."""
    return {
        "session_id": session.session_id,
        "message_count": session.message_count,
        "conversation_history": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in session.conversation_history
        ],
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
    }

class SessionManager:
    """Manages in-memory user sessions and conversation history."""

    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, SessionData] = {}
        logger.info("SessionManager initialized")

    def create_session(self, session_id: str) -> SessionData:
        """Create a new session."""
        if session_id in self.sessions:
            return self.sessions[session_id]

        session = SessionData(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve an existing session."""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to the session's conversation history."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Message dropped because session {session_id} was not found")
            return False

        message = Message(role=role, content=content)
        session.conversation_history.append(message)
        session.message_count += 1
        session.last_activity = datetime.now()
        return True

    def get_conversation_history(self, session_id: str) -> list:
        """Get formatted conversation history for the session."""
        session = self.get_session(session_id)
        if not session:
            return []
        return session.conversation_history

    def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that have exceeded the timeout."""
        current_time = datetime.now()
        timeout_seconds = settings.SESSION_TIMEOUT
        expired_sessions = []


        for session_id, session in self.sessions.items():
            elapsed = (current_time - session.last_activity).total_seconds()
            if elapsed > timeout_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.close_session(session_id)
            logger.info(f"Cleaned up expired session: {session_id}")

        return len(expired_sessions)

    def get_active_sessions_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)


# Global session manager instance
session_manager = SessionManager()
