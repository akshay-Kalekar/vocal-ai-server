## WebSocket Setup Guide

### Prerequisites
1. **Ollama installed and running** - Download from [ollama.ai](https://ollama.ai)
2. **Python 3.8+** with FastAPI and dependencies
3. **Model pulled in Ollama** - e.g., `ollama pull llama2`

### Quick Start

#### 1. Copy environment template
```bash
cp .env.example .env
```

#### 2. Install/update dependencies
```bash
pip install fastapi uvicorn httpx websockets python-dotenv pydantic
```

#### 3. Start Ollama (if not already running)
```bash
ollama serve
```

In a separate terminal, pull a model if needed:
```bash
ollama pull llama2
```

#### 4. Start the backend server
```bash
cd /home/akshay/Desktop/Low\ Cost\ Voice\ Agent/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
✓ LLM is available and ready
Uvicorn running on http://0.0.0.0:8000
```

#### 5. Test the WebSocket in another terminal
```bash
python test_client.py
```

Then type messages:
```
You: Hello, how are you?
→ Message sent
Agent: I'm doing well, thank you for asking!
(Message 1 in conversation)

You: What's your name?
→ Message sent
Agent: I'm an AI assistant created by Ollama.
(Message 2 in conversation)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health check with LLM status |
| `/ws/{session_id}` | WebSocket | Chat endpoint with session tracking |
| `/docs` | GET | Swagger UI interactive docs |

### WebSocket Protocol

**Connect to**: `ws://localhost:8000/ws/unique-session-id`

**Send (JSON)**:
```json
{
  "text": "Your message here",
  "session_id": "unique-session-id"
}
```

**Receive (JSON)**:
```json
{
  "response": "AI response here",
  "session_id": "unique-session-id",
  "timestamp": "2026-04-09T10:30:45.123456",
  "message_count": 1
}
```

**Error Response**:
```json
{
  "error": "Error description",
  "session_id": "unique-session-id",
  "timestamp": "2026-04-09T10:30:45.123456"
}
```

### Configuration (.env)

- `OLLAMA_URL`: URL of Ollama server (default: `http://localhost:11434`)
- `MODEL_NAME`: Model to use (default: `llama2`)
- `SESSION_TIMEOUT`: Session timeout in seconds (default: `3600`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Project Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration management
├── test_client.py         # WebSocket test client
├── .env.example           # Environment variables template
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models for validation
├── services/
│   ├── __init__.py
│   ├── session_manager.py # Session and conversation tracking
│   └── llm_service.py     # LLM communication via Ollama
└── routes/
    ├── __init__.py
    └── websocket.py       # WebSocket endpoint handler
```

### Troubleshooting

**Error: "Cannot connect to LLM"**
- Make sure Ollama is running: `ollama serve`
- Check OLLAMA_URL in `.env` matches your setup
- Try `curl http://localhost:11434/api/tags` to verify connectivity

**Error: "Empty response from LLM"**
- Model may be generating slowly
- Check model is properly loaded: `ollama list`
- Try a simpler model: `ollama pull mistral`

**WebSocket connection refused**
- Backend not running: `uvicorn main:app --reload`
- Check port 8000 is available or use different port

**Session timeout issues**
- Increase `SESSION_TIMEOUT` in `.env` if conversations are being cleared

### Features Implemented

✓ WebSocket real-time bidirectional communication
✓ In-memory session management with conversation history
✓ Local LLM integration via Ollama
✓ Pydantic validation for all messages
✓ Error handling and fallback responses
✓ Logging and health checks
✓ CORS enabled for frontend integration
✓ Architecture ready for STT/TTS expansion

### Next Steps

1. **Add voice support**: Integrate Whisper (STT) and Coqui TTS
2. **Persistent storage**: Replace in-memory sessions with database
3. **Scaling**: Move to Redis/message queues for multi-worker setup
4. **Frontend**: Build web UI with WebSocket client
5. **Monitoring**: Add Prometheus metrics and structured logging
