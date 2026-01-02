"""
Pipeline Orchestrator for PrecisionVoice.
Coordinates transcription and diarization in parallel with progress updates.
"""
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Any

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse, TranscriptSegment
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService

logger = logging.getLogger(__name__)
settings = get_settings()

class PipelineOrchestrator:
    """
    Coordinates the AI pipeline with real-time status updates:
    1. Audio -> WAV (Noise Reduction)
    2. Whisper (Transcribe) + Pyannote (Diarize) in parallel
    3. Alignment (Matching Algorithm)
    4. Generate outputs (TXT, SRT)
    """

    @classmethod
    async def process_audio_stream(
        cls, 
        wav_path: Path, 
        duration: float
    ) -> AsyncGenerator[str, None]:
        """
        Run the full processing pipeline and yield progress status.
        Yields JSON strings for SSE.
        """
        start_time = time.time()
        
        # Step 1: Noise Reduction (Already handled by AudioProcessor, but we report it here)
        yield json.dumps({"status": "processing", "message": "Noise reduction applied", "progress": 20})
        
        # Step 2: Parallel Whisper and Pyannote
        yield json.dumps({"status": "processing", "message": "Transcription & Diarization starting...", "progress": 40})
        
        transcription_task = TranscriptionService.transcribe_async(wav_path)
        diarization_task = DiarizationService.diarize_async(wav_path)
        
        try:
            # We can't easily get partial progress from these black-box models without deeper integration,
            # so we wait for both and then jump to 80%
            word_timestamps, speaker_segments = await asyncio.gather(
                transcription_task,
                diarization_task,
                return_exceptions=False
            )
            yield json.dumps({"status": "processing", "message": "AI models finished processing", "progress": 80})
        except Exception as e:
            logger.exception("Parallel task failed")
            yield json.dumps({"status": "error", "message": f"AI processing failed: {str(e)}"})
            return

        # Step 3: Precision alignment
        yield json.dumps({"status": "processing", "message": "Aligning speakers with text...", "progress": 90})
        aligned_segments = AlignmentService.align_precision(word_timestamps, speaker_segments)
        
        # Count unique speakers
        speakers = set(seg.speaker for seg in aligned_segments)
        
        # Step 4: Generate output files
        yield json.dumps({"status": "processing", "message": "Generating export files...", "progress": 95})
        base_filename = wav_path.stem.replace("_processed", "")
        txt_path, srt_path = AlignmentService.generate_outputs(aligned_segments, base_filename)
        
        processing_time = time.time() - start_time
        
        # Final result
        result = TranscriptionResponse(
            success=True,
            message="Transcription completed successfully",
            segments=aligned_segments,
            duration=duration,
            num_speakers=len(speakers),
            processing_time=round(processing_time, 2),
            download_txt=f"/api/download/{txt_path.name}",
            download_srt=f"/api/download/{srt_path.name}"
        )
        
        yield json.dumps({
            "status": "completed", 
            "message": "Processing complete", 
            "progress": 100,
            "result": result.model_dump()
        })
