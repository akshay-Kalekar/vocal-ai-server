"""Simple WebSocket test client for testing the agent."""

import asyncio
import json
import websockets
import sys


async def test_websocket(session_id: str = "test-session-1"):
    """Connect to WebSocket and send test messages.

    Args:
        session_id: Unique session identifier
    """
    uri = f"ws://localhost:8000/ws/{session_id}"
    
    print(f"Connecting to {uri}")
    print("Type 'exit' to quit\n")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to WebSocket!\n")

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    print("Closing connection...")
                    break
                if not user_input:
                    continue
                message = {"text": user_input, "session_id": session_id}
                await websocket.send(json.dumps(message))
                print("→ Message sent")

                # Handle streaming responses
                try:
                    partial = ""
                    message_count = None
                    while True:
                        response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                        data = json.loads(response)
                        if "error" in data:
                            print(f"✗ Error: {data['error']}\n")
                            break
                        if data.get("stream"):
                            # Print only new part of the response
                            new_text = data["response"][len(partial):]
                            print(new_text, end="", flush=True)
                            partial = data["response"]
                            message_count = data.get("message_count")
                        else:
                            # Final message (if any non-stream payload is sent)
                            break
                    if partial:
                        print(f"\n(Message {message_count} in conversation)\n")
                except asyncio.TimeoutError:
                    print("✗ Timeout waiting for response (LLM may be slow or unavailable)\n")

    except ConnectionRefusedError:
        print("✗ Cannot connect to server. Is it running? (uvicorn main:app --reload)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "test-session-1"
    asyncio.run(test_websocket(session_id))
