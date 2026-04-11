# Low Cost Voice Agent — Backend

FastAPI service that powers a text (and optional voice) assistant: it streams replies from a **local LLM via Ollama**, keeps **in-memory conversation history** per WebSocket session, and can stream **TTS audio** back over the same socket when enabled.

---

## How it works

1. **HTTP** — The app exposes a small REST surface (`/`, `/health`) for sanity checks and LLM reachability.
2. **WebSocket** — All chat traffic goes through **`/ws/{session_id}`**. The `session_id` in the URL is the conversation key; the server creates or resumes that session when the socket connects.
3. **LLM** — Each user message is appended to the session, then the server calls Ollama’s **`/api/generate`** and streams tokens. You receive many JSON frames with `stream: true` and a growing `response` field, then one final frame with `stream: false`.
4. **TTS (optional)** — If `TTS_ENABLED=true`, the server runs **Qwen TTS** on the assistant text: it produces **mono PCM**, then either wraps **WAV** or encodes **Opus** for the wire. The utterance is **fully synthesized before** `audio_chunk` frames are sent (not streamed sample-by-sample from the TTS model).
5. **Welcome audio** — With TTS on, the server may send a **welcome** utterance (`context: "welcome"`) right after `session_ready`, before you send any user text.

---

## Quick start

1. Copy [`.env.example`](./.env.example) to `.env` and adjust variables.
2. Run **Ollama** and pull a model (see `.env.example` for `MODEL_NAME`).
3. Install dependencies (use a virtual environment on Debian/Ubuntu systems that block global `pip`):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install fastapi uvicorn httpx websockets python-dotenv pydantic
   ```

   For **Opus** TTS over the socket (`TTS_AUDIO_CODEC=opus`, default): also `pip install opuslib-next` and install **libopus** on the system (e.g. `libopus0`).

   For **TTS**: install the **Qwen TTS** stack your project uses (`qwen_tts`, PyTorch, etc.) per your environment.

4. Start the API:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Smoke-test the WebSocket:

   ```bash
   python test_client.py
   ```

---

## REST endpoints

| Endpoint | Method | Purpose |
|-----------|--------|---------|
| `/`       | GET    | Short JSON banner + API version |
| `/health` | GET    | Health + `llm_available`, model name, Ollama URL |
| `/docs`   | GET    | Swagger UI |

CORS is configured with **wide open** origins (`*`) for local frontend development. Tighten this before production.

---

## Frontend: connecting over WebSocket

### URL

Use the same host and port as the HTTP API, with scheme **`ws`** or **`wss`**:

- Local: `ws://localhost:8000/ws/<session_id>`
- Production: `wss://your-domain.com/ws/<session_id>`

Pick a stable **`session_id`** per user or per conversation (UUID string is fine). It must match the path; an optional `session_id` field in the JSON body is not required for routing.

### Browser example

```javascript
const sessionId = crypto.randomUUID();
const ws = new WebSocket(`ws://${location.hostname}:8000/ws/${sessionId}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "session_ready") {
    // msg.tts_enabled, msg.tts_codec ("opus" | "wav")
    return;
  }

  if (msg.stream === true && msg.response !== undefined) {
    // Streaming assistant text — append delta: msg.response is the full text so far
    return;
  }

  if (msg.stream === false && msg.response !== undefined) {
    // Final text for this turn; if msg.tts_follows, audio frames come next
    return;
  }

  if (msg.type === "audio_start" || msg.type === "audio_chunk" || msg.type === "audio_end") {
    // See “TTS audio protocol” below
    return;
  }

  if (msg.type === "audio_error") {
    // msg.context: "welcome" | "reply"
    return;
  }

  if (msg.error) {
    // Validation / server error
    return;
  }
};

// Send a user turn (after open)
ws.send(JSON.stringify({ text: "Hello!" }));
```

### Sending user input

Send **one JSON object per user message** (text frame, not binary):

```json
{
  "text": "Your message here"
}
```

`session_id` inside the object is optional and ignored for the socket path; the path segment is authoritative.

---

## Assistant text streaming protocol

All assistant replies use the same shape (Pydantic `AgentResponse` plus extras):

- **Streaming chunks** (many frames): `stream: true`, `response` = **full assistant text so far** (not a delta). The client should replace the visible reply with `response` or compute the suffix compared to the previous length.
- **Final frame**: `stream: false`, `response` = final full text, `message_count` updated, optional `session` snapshot, and **`tts_follows`: `true`** if TTS audio for this reply will be sent next.

Errors use `ErrorResponse`-style JSON: top-level `error` string and optional `session_id`.

---

## TTS audio protocol (when `TTS_ENABLED=true`)

After `session_ready`, you may receive a **welcome** sequence. After each turn, if `tts_follows` is true, you receive a **reply** sequence.

For both, frames are JSON text messages with a `type` field:

1. **`audio_start`** — `context`: `"welcome"` | `"reply"`, `format`: `"opus"` | `"wav"`, optional `sample_rate`, `opus_frame_samples` (Opus), short `text` preview.
2. **`audio_chunk`** — Base64 in **`data`**, ordered by **`sequence`** (0 … `total_chunks` − 1).
   - **WAV**: concatenate decoded bytes to rebuild a full `.wav` file; `sample_rate` is the PCM rate inside the WAV.
   - **Opus**: each chunk is one **Opus packet**; decode at **48 kHz** mono with **960 samples** (20 ms) per packet (`opus_frame_samples`). The server resamples TTS PCM to 48 kHz before encoding.
3. **`audio_end`** — Same `format` / `sample_rate` as chunks; end of this utterance.
4. **`audio_error`** — TTS failed; `context` and `error` string.

Check **`session_ready.tts_codec`** so the UI knows whether to treat chunks as WAV or Opus. If Opus libraries are missing on the server, it falls back to **`wav`** automatically.

### Browser playback notes

- **WAV**: After `audio_end`, you can build a `Blob` and use `URL.createObjectURL` with an `<audio>` element, or decode with Web Audio.
- **Raw Opus packets** are not a single `.opus` file or Ogg container; playing them in the browser usually means **WASM Opus** (e.g. `opus-decoder`) or **WebCodecs** where supported, or ask the operator to set `TTS_AUDIO_CODEC=wav` for simpler `<audio>` playback at the cost of larger payloads.

---

## Configuration

See [`.env.example`](./.env.example) for variables. Important groups:

- **LLM**: `OLLAMA_URL`, `MODEL_NAME`
- **TTS**: `TTS_ENABLED`, `TTS_MODEL_NAME`, timeouts, `TTS_AUDIO_CODEC` (`opus` | `wav`), etc.
- **Server**: `LOG_LEVEL`, `SESSION_TIMEOUT`

---

## Project layout (high level)

| Path | Role |
|------|------|
| `main.py` | FastAPI app, CORS, lifespan hooks |
| `config.py` | Environment-based settings |
| `routes/websocket.py` | Chat + TTS streaming |
| `services/llm_service.py` | Ollama streaming client |
| `services/tts_service.py` | Qwen TTS → PCM (+ optional WAV wrap) |
| `services/opus_encoder.py` | PCM → Opus packets |
| `services/session_manager.py` | Per-session history |
| `models/schemas.py` | Pydantic message types |
| `test_client.py` | CLI WebSocket test (text + optional TTS save) |

---

## Troubleshooting

- **`/health` shows `llm_available: false`** — Ollama not running or wrong `OLLAMA_URL`.
- **No audio** — `TTS_ENABLED` must be `true`; Qwen TTS dependencies must load; for Opus, `libopus` + `opuslib-next` must be available.
- **WebSocket fails from the browser** — Use `wss://` when the page is served over HTTPS; ensure the API host/port and firewall allow WebSocket upgrades.

For a step-by-step local setup (Ollama, env file, `uvicorn`), you can still use [WEBSOCKET_SETUP.md](./WEBSOCKET_SETUP.md); this README is the canonical description of the **current** protocol and frontend integration.
