import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.websocket import router as websocket_router
from services.llm_service import llm_service
from services.tts_service import tts_service

logger = settings.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
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
    return {"message": "Low Cost Voice Agent Backend", "version": settings.API_VERSION}



@app.get("/health")
async def health_check():
    """Health check endpoint. Includes LLM status."""
    llm_available = await llm_service.check_connection()
    return {
        "status": "healthy",
        "llm_available": llm_available,
        "model": settings.MODEL_NAME,
        "ollama_url": settings.OLLAMA_URL,
    }



@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 50)
    logger.info("Low Cost Voice Agent Backend Starting")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Session Timeout: {settings.SESSION_TIMEOUT}s")
    logger.info("=" * 50)

    # Check LLM availability on startup
    if await llm_service.check_connection():
        logger.info("✓ LLM is available and ready")
    else:
        logger.warning("⚠ LLM is not reachable. Make sure Ollama is running.")

    if settings.TTS_ENABLED:
        if await tts_service.check_connection():
            logger.info("✓ Qwen TTS is available and ready")
        else:
            logger.warning("⚠ Qwen TTS is enabled but not reachable.")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Low Cost Voice Agent Backend Shutting Down")

