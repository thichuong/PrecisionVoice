"""
API routes for the transcription service.
"""
import time
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse, ErrorResponse, HealthResponse
from app.services.audio_processor import AudioProcessor, AudioProcessingError
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService
from app.services.orchestrator import PipelineOrchestrator

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


@router.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Upload and transcribe an audio file with speaker diarization.
    
    Supported formats: mp3, wav, m4a, ogg, flac, webm
    Maximum file size: 100MB
    
    Returns transcription with speaker labels and download URLs.
    """
    start_time = time.time()
    wav_path = None
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate and process audio
        try:
            AudioProcessor.validate_file(file.filename or "audio.wav", len(file_content))
        except AudioProcessingError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Save and convert to WAV
        wav_path, duration = await AudioProcessor.process_upload(
            file_content, 
            file.filename or "audio.wav"
        )
        
        # Run orchestrated pipeline (Whisper + Pyannote in parallel -> Alignment)
        logger.info("Executing orchestrated pipeline...")
        response = await PipelineOrchestrator.process_audio(wav_path, duration)
        
        # Schedule cleanup in background
        background_tasks.add_task(cleanup_files, wav_path)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed")
        # Cleanup on error
        if wav_path and wav_path.exists():
            background_tasks.add_task(cleanup_files, wav_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("/api/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
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
    
    # Schedule cleanup after download (give some time for download to complete)
    # Note: In production, you might want a separate cleanup job
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type
    )


async def cleanup_files(*paths: Path):
    """Background task to cleanup temporary files."""
    import asyncio
    
    # Wait a bit before cleanup to ensure files are not in use
    await asyncio.sleep(5)
    
    await AudioProcessor.cleanup_files(*paths)
