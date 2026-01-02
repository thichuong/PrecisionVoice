"""
Speaker diarization service using pyannote.audio.
Identifies speaker turns in audio files.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""
    start: float
    end: float
    speaker: str


class DiarizationService:
    """
    Service for speaker diarization using pyannote.audio.
    Implements lazy loading to avoid memory overhead at startup.
    """
    
    _instance: Optional["DiarizationService"] = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_pipeline(cls):
        """
        Get or load the diarization pipeline (lazy loading with caching).
        
        Returns:
            Loaded pyannote Pipeline
        """
        if cls._pipeline is None:
            # Import here to avoid loading if not used
            from pyannote.audio import Pipeline
            
            hf_token = settings.hf_token
            if not hf_token:
                raise ValueError(
                    "HuggingFace token required for pyannote.audio. "
                    "Set HF_TOKEN in your environment or .env file."
                )
            
            logger.info(f"Loading diarization pipeline: {settings.diarization_model}")
            
            # Use 'token' parameter (use_auth_token is deprecated)
            cls._pipeline = Pipeline.from_pretrained(
                settings.diarization_model,
                token=hf_token
            )
            
            # Move to GPU if available
            device = torch.device(settings.resolved_device)
            if device.type == "cuda":
                cls._pipeline = cls._pipeline.to(device)
                logger.info("Diarization pipeline moved to GPU")
            
            logger.info("Diarization pipeline loaded successfully")
        
        return cls._pipeline
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if pipeline is loaded."""
        return cls._pipeline is not None
    
    @classmethod
    def diarize(
        cls,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to WAV audio file
            num_speakers: Exact number of speakers (None for auto-detect)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            
        Returns:
            List of SpeakerSegment with speaker labels
        """
        pipeline = cls.get_pipeline()
        
        logger.info(f"Diarizing: {audio_path}")
        
        # Build parameters
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers
        
        # Run diarization
        diarization = pipeline(str(audio_path), **params)
        
        # Convert to segments
        segments = []
        speaker_map = {}  # Map SPEAKER_XX to Speaker 1, 2, etc.
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Create readable speaker label
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
            
            segments.append(SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker_map[speaker]
            ))
        
        logger.info(f"Diarization complete: {len(segments)} turns, {len(speaker_map)} speakers")
        
        return segments
    
    @classmethod
    async def diarize_async(
        cls,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> List[SpeakerSegment]:
        """
        Async wrapper for diarization (runs in thread pool).
        
        Args:
            audio_path: Path to WAV audio file
            num_speakers: Exact number of speakers
            min_speakers: Minimum speakers
            max_speakers: Maximum speakers
            
        Returns:
            List of SpeakerSegment
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.diarize(audio_path, num_speakers, min_speakers, max_speakers)
        )
    
    @classmethod
    def preload_pipeline(cls) -> None:
        """Preload the pipeline during startup."""
        try:
            cls.get_pipeline()
        except Exception as e:
            logger.warning(f"Failed to preload diarization pipeline: {e}")
            # Don't raise - diarization is optional, app can work without it
