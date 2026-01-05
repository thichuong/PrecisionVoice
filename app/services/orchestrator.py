"""
Pipeline Orchestrator for PrecisionVoice.
Coordinates transcription and diarization in parallel.
"""
import time
import asyncio
import logging
from pathlib import Path

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService

logger = logging.getLogger(__name__)
settings = get_settings()

class PipelineOrchestrator:
    """
    Coordinates the AI pipeline with detailed server-side logging:
    1. Audio -> Vocal Separation (MDX-Net) -> 16kHz WAV
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
        Run the full processing pipeline and return the final response.
        Each step is logged for server-side monitoring.
        """
        start_time = time.time()
        
        # Step 1: Pre-processing (Vocal Separation + Noise Reduction)
        logger.info(f"[Step 1/4] Audio pre-processing completed (MDX-Net: {settings.enable_vocal_separation}, Denoise: {settings.enable_noise_reduction})")
        
        # Step 2: AI Processing (Transcription & Diarization)
        logger.info(f"[Step 2/4] Starting AI models (Whisper + Pyannote) for: {wav_path.name}")
        
        transcription_task = TranscriptionService.transcribe_async(wav_path)
        diarization_task = DiarizationService.diarize_async(wav_path)
        
        try:
            word_timestamps, speaker_segments = await asyncio.gather(
                transcription_task,
                diarization_task,
                return_exceptions=False
            )
            logger.info(f"AI models processing completed: {len(word_timestamps)} words, {len(speaker_segments)} segments")
        except Exception as e:
            logger.exception("Parallel task failed")
            raise

        # Step 3: Precision Alignment
        logger.info("[Step 3/4] Aligning words with speaker turns...")
        aligned_segments = AlignmentService.align_precision(word_timestamps, speaker_segments)
        
        # Count unique speakers
        speakers = set(seg.speaker for seg in aligned_segments)
        
        # Step 4: Export Generation
        logger.info("[Step 4/4] Generating export files (TXT, SRT)...")
        base_filename = wav_path.stem.replace("_processed", "")
        txt_path, srt_path = AlignmentService.generate_outputs(aligned_segments, base_filename)
        
        processing_time = time.time() - start_time
        logger.info(f"Pipeline complete for {wav_path.name} in {processing_time:.2f}s")
        
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
