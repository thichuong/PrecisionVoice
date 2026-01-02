"""
PrecisionVoice - Speech-to-Text & Speaker Diarization Application

Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.api.routes import router
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Preloads models on startup for faster first request.
    """
    logger.info("Starting PrecisionVoice application...")
    logger.info(f"Device: {settings.resolved_device}")
    logger.info(f"Whisper model: {settings.whisper_model}")
    logger.info(f"Diarization model: {settings.diarization_model}")
    
    # Preload models (optional - can be disabled for faster startup)
    try:
        logger.info("Preloading Whisper model...")
        TranscriptionService.preload_model()
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}")
    
    try:
        if settings.hf_token:
            logger.info("Preloading diarization pipeline...")
            DiarizationService.preload_pipeline()
        else:
            logger.warning("HF_TOKEN not set, diarization will not be available")
    except Exception as e:
        logger.warning(f"Diarization preload failed (will try again on first use): {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    logger.info("Shutting down PrecisionVoice application...")


# Create FastAPI app
app = FastAPI(
    title="PrecisionVoice",
    description="Speech-to-Text and Speaker Diarization API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "max_upload_mb": settings.max_upload_size_mb,
            "allowed_formats": ", ".join(settings.allowed_extensions)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
