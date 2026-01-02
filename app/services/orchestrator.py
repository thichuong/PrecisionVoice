"""
Pipeline Orchestrator for PrecisionVoice.
Coordinates transcription and diarization in parallel.
"""
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse, TranscriptSegment
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService

logger = logging.getLogger(__name__)
settings = get_settings()

class PipelineOrchestrator:
    """
    Coordinates the AI pipeline:
    1. Audio -> WAV (done by AudioProcessor before calling this)
    2. Whisper (Transcribe) + Pyannote (Diarize) in parallel
    3. Alignment (Matching Algorithm)
    4. Generate outputs (TXT, SRT)
    """

    @classmethod
    async def process_audio(
        cls, 
        wav_path: Path, 
        duration: float
    ) -> TranscriptionResponse:
        """
        Run the full processing pipeline on a WAV file.
        """
        start_time = time.time()
        
        # Step 3a & 3b: Run Whisper and Pyannote in parallel
        logger.info(f"Starting parallel processing for: {wav_path.name}")
        
        transcription_task = TranscriptionService.transcribe_async(wav_path)
        diarization_task = DiarizationService.diarize_async(wav_path)
        
        # Execute in parallel
        try:
            word_timestamps, speaker_segments = await asyncio.gather(
                transcription_task,
                diarization_task,
                return_exceptions=False
            )
            logger.info(f"Parallel processing complete: {len(word_timestamps)} words, {len(speaker_segments)} speaker segments")
        except Exception as e:
            logger.exception("Parallel task failed")
            # Fallback logic if one fails? 
            # If diarization fails, we can still return transcription with a single speaker
            if 'word_timestamps' not in locals():
                raise # Critical failure
            speaker_segments = []
            logger.warning("Continuing with empty speaker segments due to diarization failure")

        # Step 3c & 3d: Precision alignment (word-center-based)
        logger.info("Running precision alignment...")
        aligned_segments = AlignmentService.align_precision(word_timestamps, speaker_segments)
        
        # Count unique speakers
        speakers = set(seg.speaker for seg in aligned_segments)
        
        # Generate output files
        base_filename = wav_path.stem.replace("_processed", "")
        txt_path, srt_path = AlignmentService.generate_outputs(aligned_segments, base_filename)
        
        processing_time = time.time() - start_time
        
        return TranscriptionResponse(
            success=True,
            message="Transcription completed successfully",
            segments=aligned_segments,
            duration=duration,
            num_speakers=len(speakers),
            processing_time=round(processing_time, 2),
            download_txt=f"/api/download/{txt_path.name}",
            download_srt=f"/api/download/{srt_path.name}"
        )
