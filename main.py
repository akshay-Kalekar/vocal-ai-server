import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.websocket import router as websocket_router
from services.llm_service import llm_service
from services.tts_service import tts_service
import config

# Configure logging
log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Keep dependency logs concise unless explicitly debugging.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Low Cost Voice Agent - Real-time chat with local LLM",
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebSocket routes
app.include_router(websocket_router)


@app.get("/")
async def root():
    return {"message": "Low Cost Voice Agent Backend", "version": config.API_VERSION}


@app.get("/health")
async def health_check():
    """Health check endpoint. Includes LLM status."""
    llm_available = await llm_service.check_connection()
    return {
        "status": "healthy",
        "llm_available": llm_available,
        "model": config.MODEL_NAME,
        "ollama_url": config.OLLAMA_URL,
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 50)
    logger.info("Low Cost Voice Agent Backend Starting")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Ollama URL: {config.OLLAMA_URL}")
    logger.info(f"Session Timeout: {config.SESSION_TIMEOUT}s")
    logger.info("=" * 50)

    # Check LLM availability on startup
    if await llm_service.check_connection():
        logger.info("✓ LLM is available and ready")
    else:
        logger.warning("⚠ LLM is not reachable. Make sure Ollama is running.")

    if config.TTS_ENABLED:
        if await tts_service.check_connection():
            logger.info("✓ Qwen TTS is available and ready")
        else:
            logger.warning("⚠ Qwen TTS is enabled but not reachable.")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Low Cost Voice Agent Backend Shutting Down")
