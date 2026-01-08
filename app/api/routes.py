"""
API routes for the transcription service.
"""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse, HealthResponse
from app.services.audio_processor import AudioProcessor, AudioProcessingError
from app.services.transcription import TranscriptionService, AVAILABLE_MODELS
from app.services.diarization import DiarizationService
from app.services.processor import Processor

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=TranscriptionService.is_loaded() and DiarizationService.is_loaded(),
        device=settings.resolved_device
    )


@router.get("/api/models")
async def get_models():
    """Get available Whisper models."""
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "default": settings.default_whisper_model
    }


@router.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="EraX-WoW-Turbo", description="Whisper model to use"),
    language: str = Form(default="vi", description="Language code"),
    merge_segments: bool = Form(default=True, description="Merge consecutive speaker segments"),
):
    """
    Upload and transcribe an audio file.
    
    Uses diarize-first workflow:
    1. Diarization to identify speakers
    2. Transcribe each speaker segment
    3. Return combined result
    """
    upload_path = None
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate
        try:
            AudioProcessor.validate_file(file.filename or "audio.wav", len(file_content))
        except AudioProcessingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Save upload
        upload_path = await AudioProcessor.save_upload(file_content, file.filename or "audio.wav")
        
        # Process with new workflow
        logger.info(f"Processing audio with model={model}, language={language}")
        result = await Processor.process_audio(
            audio_path=upload_path,
            model_name=model,
            language=language,
            merge_segments=merge_segments,
        )
        
        # Save output files
        txt_filename = f"{upload_path.stem}_transcript.txt"
        srt_filename = f"{upload_path.stem}_transcript.srt"
        
        txt_path = settings.processed_dir / txt_filename
        srt_path = settings.processed_dir / srt_filename
        
        txt_path.write_text(result.txt_content, encoding="utf-8")
        srt_path.write_text(result.srt_content, encoding="utf-8")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_files, upload_path)
        
        # Build response
        return TranscriptionResponse(
            success=True,
            segments=[
                {
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "text": seg.text
                }
                for seg in result.segments
            ],
            speaker_count=result.speaker_count,
            duration=result.duration,
            processing_time=result.processing_time,
            download_txt=f"/api/download/{txt_filename}",
            download_srt=f"/api/download/{srt_filename}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed")
        if upload_path and upload_path.exists():
            background_tasks.add_task(cleanup_files, upload_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated transcript file.
    Supports: .txt, .srt files
    """
    # Security: only allow specific extensions and no path traversal
    if not filename.endswith(('.txt', '.srt')) or '/' in filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = settings.processed_dir / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_type = "text/plain" if filename.endswith('.txt') else "application/x-subrip"
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type
    )


async def cleanup_files(*paths: Path):
    """Background task to cleanup temporary files."""
    import asyncio
    
    # Wait a bit before cleanup
    await asyncio.sleep(5)
    
    await AudioProcessor.cleanup_files(*paths)
