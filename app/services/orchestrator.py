"""
Pipeline Orchestrator for PrecisionVoice.
Coordinates transcription and diarization in parallel.
Integrates Silero VAD for hallucination prevention.
"""
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService
from app.services.alignment import AlignmentService
from app.services.vad import VADService, SpeechSegment

logger = logging.getLogger(__name__)
settings = get_settings()

class PipelineOrchestrator:
    """
    Coordinates the AI pipeline with detailed server-side logging:
    1. Audio -> Vocal Separation (MDX-Net) -> 16kHz WAV
    2. Silero VAD (Filter silence) - NEW
    3. Whisper (Transcribe filtered) + Pyannote (Diarize original) in parallel
    4. Alignment (Matching Algorithm with timestamp reconstruction)
    5. Generate outputs (TXT, SRT)
    """

    @classmethod
    async def process_audio(
        cls, 
        wav_path: Path,
        vad_filtered_path: Path,
        duration: float,
        original_segments: List[SpeechSegment],
        adjusted_segments: List[SpeechSegment]
    ) -> TranscriptionResponse:
        """
        Run the full processing pipeline and return the final response.
        Each step is logged for server-side monitoring.
        
        Args:
            wav_path: Full audio (for diarization)
            vad_filtered_path: VAD-filtered audio (for transcription)
            duration: Audio duration in seconds
            original_segments: Speech segments with original timestamps
            adjusted_segments: Speech segments with adjusted timestamps
        """
        start_time = time.time()
        
        # Step 1: Pre-processing info
        logger.info(f"[Step 1/5] Audio pre-processing completed (MDX-Net: {settings.enable_vocal_separation}, SpeechBrain: {settings.enable_speech_enhancement})")
        
        # Step 2: VAD info
        vad_enabled = settings.enable_silero_vad and original_segments
        if vad_enabled:
            logger.info(f"[Step 2/5] Silero VAD: {len(original_segments)} speech segments detected, silence filtered")
        else:
            logger.info(f"[Step 2/5] Silero VAD: Disabled or no filtering applied")
        
        # Step 3: AI Processing (Transcription & Diarization in parallel)
        logger.info(f"[Step 3/5] Starting AI models - Whisper on VAD-filtered, Pyannote on full audio")
        
        # CRITICAL: Whisper uses VAD-filtered audio, Pyannote uses original full audio
        transcription_task = TranscriptionService.transcribe_async(vad_filtered_path)
        diarization_task = DiarizationService.diarize_async(wav_path)
        
        try:
            word_timestamps, speaker_segments = await asyncio.gather(
                transcription_task,
                diarization_task,
                return_exceptions=False
            )
            logger.info(f"AI models completed: {len(word_timestamps)} words, {len(speaker_segments)} speaker turns")
        except Exception as e:
            logger.exception("Parallel task failed")
            raise
        
        # Step 4: Timestamp reconstruction (if VAD was applied)
        if vad_enabled and original_segments and adjusted_segments:
            logger.info("[Step 4/5] Reconstructing timestamps to original timeline...")
            word_timestamps = VADService.map_timestamps_to_original(
                word_timestamps,
                original_segments,
                adjusted_segments
            )
            logger.info(f"Timestamps reconstructed for {len(word_timestamps)} words")
        else:
            logger.info("[Step 4/5] No timestamp reconstruction needed")

        # Step 5: Precision Alignment
        logger.info("[Step 5/5] Aligning words with speaker turns...")
        aligned_segments = AlignmentService.align_precision(word_timestamps, speaker_segments)
        
        # Count unique speakers
        speakers = set(seg.speaker for seg in aligned_segments)
        
        # Generate export files
        logger.info("Generating export files (TXT, SRT)...")
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
