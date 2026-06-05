import json
import re
import httpx
from typing import AsyncGenerator, List
import settings
from models.schemas import Message

logger = settings.get_logger(__name__)


class LLMService:
    """Handles interactions with local LLM running on Ollama."""

    def __init__(self):
        """Initialize the LLM service.

        Args:
            base_url: Ollama server URL (e.g., http://localhost:11434)
            model_name: Model name to use (e.g., llama2, mistral, neural-chat)
        """
        self.base_url = settings.OLLAMA_URL
        self.model_name = settings.MODEL_NAME
        self.endpoint = f"{self.base_url}/api/generate"
        logger.info(f"LLMService initialized with model: {self.model_name} at {self.base_url}")


    def format_conversation_context(self, history: List[Message]) -> str:
        """Format conversation history into a context string for the model.

        Args:
            history: List of messages in the conversation

        Returns:
            Formatted string representation of conversation
        """
        context = ""
        for msg in history:
            if msg.role == "user":
                context += f"User: {msg.content}\n"
            else:
                context += f"Assistant: {msg.content}\n"
        return context

    # System instruction sent to Ollama on every request.
    # Keeps output plain-text so Piper TTS doesn't speak raw symbols.
    _SYSTEM_PROMPT: str = (
        "You are a helpful voice assistant. "
        "Respond in plain conversational sentences only. "
        "Do NOT use markdown, bullet points, numbered lists, asterisks, "
        "underscores, hash symbols, backticks, or any other formatting symbols. "
        "Never highlight or bold words. Write as if speaking aloud."
    )

    # Regex that removes common markdown / TTS-hostile symbols from a string.
    _MD_PATTERN: re.Pattern = re.compile(
        r"(\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|~{2}|\\|>\s?|[-*+]\s(?=\S)|\d+\.\s(?=\S))"
    )

    @classmethod
    def _clean_response(cls, text: str) -> str:
        """Strip markdown / formatting symbols that Piper TTS would read literally.

        Handles: **bold**, *italic*, ***bold-italic***, __underline__,
        `code`, ```blocks```, # headings, > blockquotes, - / * / + bullets,
        1. numbered lists, ~~strikethrough~~.
        """
        # Remove block-level markdown leaders (headings, blockquotes, bullets)
        cleaned = cls._MD_PATTERN.sub("", text)
        # Collapse excess whitespace left behind by removed symbols
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_prompt(self, user_input: str, conversation_history: List[Message]) -> str:
        """Build a bounded prompt to avoid runaway token usage."""
        trimmed_history = conversation_history[-4:]
        context = self.format_conversation_context(trimmed_history)
        return f"{context}User: {user_input}\nAssistant:"

    async def generate_response_stream(
        self, user_input: str, conversation_history: List[Message]
    ) -> AsyncGenerator[str, None]:
        """Yield streamed response chunks from Ollama."""
        prompt = self._build_prompt(user_input, conversation_history)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self._SYSTEM_PROMPT,
            "num_predict": 80,
            "keep_alive": "10m",
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", self.endpoint, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise Exception(f"LLM error {response.status_code}: {body.decode(errors='ignore')}")

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk  # raw text — callers decide what to clean
                        if data.get("done"):
                            break
        except Exception as e:
            logger.error(f"Streaming LLM error: {str(e)}")
            raise

    async def generate_response(self, user_input: str, conversation_history: List[Message]) -> str:
        """Return a full response assembled from streamed chunks (raw, no cleaning)."""
        chunks: List[str] = []
        async for chunk in self.generate_response_stream(user_input, conversation_history):
            chunks.append(chunk)
        return "".join(chunks).strip()

    async def check_connection(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            is_available = response.status_code == 200
            logger.info(f"LLM connection check: {'OK' if is_available else 'FAILED'}")
            return is_available
        except Exception as e:
            logger.warning(f"LLM connection check failed: {str(e)}")
            return False


# Global LLM service instance
llm_service = LLMService()
