"""Pydantic schemas for WebSocket messages and data structures."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Message(BaseModel):
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class UserMessage(BaseModel):
    """Incoming message from the client."""
    text: str = Field(..., min_length=1, description="User's text input")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")


class AgentResponse(BaseModel):
    """Response from the agent to send back to client."""
    response: str = Field(..., description="AI-generated response text")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    message_count: int = Field(default=0, description="Total messages in conversation")


class SessionData(BaseModel):
    """Session metadata and conversation history."""
    session_id: str
    conversation_history: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = 0


class ErrorResponse(BaseModel):
    """Error response to send to client."""
    error: str
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
