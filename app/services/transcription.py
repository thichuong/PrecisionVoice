"""
Transcription service using faster-whisper.
Loads the suzii/vi-whisper-large-v3-turbo-v1-ct2 model for Vietnamese STT.
Returns word-level timestamps for precision alignment.
"""
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from faster_whisper import WhisperModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class WordTimestamp:
    """A single word with precise timestamp."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegmentRaw:
    """Raw segment from Whisper transcription with word-level data."""
    start: float
    end: float
    text: str
    words: List[WordTimestamp]


class TranscriptionService:
    """
    Service for speech-to-text transcription using faster-whisper.
    Implements singleton pattern for model caching.
    Returns word-level timestamps for precision speaker alignment.
    """
    
    _instance: Optional["TranscriptionService"] = None
    _model: Optional[WhisperModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls) -> WhisperModel:
        """
        Get or load the Whisper model (lazy loading with caching).
        
        Returns:
            Loaded WhisperModel instance
        """
        if cls._model is None:
            logger.info(f"Loading Whisper model: {settings.whisper_model}")
            logger.info(f"Device: {settings.resolved_device}, Compute type: {settings.resolved_compute_type}")
            
            cls._model = WhisperModel(
                settings.whisper_model,
                device=settings.resolved_device,
                compute_type=settings.resolved_compute_type,
                download_root=None,  # Use default HF cache
            )
            
            logger.info("Whisper model loaded successfully")
        
        return cls._model
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is loaded."""
        return cls._model is not None
    
    @classmethod
    def transcribe(
        cls,
        audio_path: Path,
        language: str = "vi",
        initial_prompt: Optional[str] = None
    ) -> List[WordTimestamp]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to WAV audio file
            language: Language code (default: Vietnamese)
            initial_prompt: Optional prompt for context
            
        Returns:
            List of WordTimestamp with precise timing for each word
        """
        model = cls.get_model()
        
        logger.info(f"Transcribing: {audio_path}")
        
        # Run transcription with word timestamps - CRITICAL for precision alignment
        segments_generator, info = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=True,  # CRITICAL: Enable word-level timestamps
            vad_filter=False,  # Disabled - was filtering out too much audio
            beam_size=5,
            best_of=5,
        )
        
        # Extract all words with timestamps
        all_words = []
        segment_count = 0
        
        for segment in segments_generator:
            segment_count += 1
            if segment.words:
                for word in segment.words:
                    all_words.append(WordTimestamp(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end
                    ))
        
        logger.info(f"Transcription complete: {segment_count} segments, {len(all_words)} words, detected language: {info.language}")
        
        return all_words
    
    @classmethod
    async def transcribe_async(
        cls,
        audio_path: Path,
        language: str = "vi",
        initial_prompt: Optional[str] = None
    ) -> List[WordTimestamp]:
        """
        Async wrapper for transcription (runs in thread pool).
        
        Args:
            audio_path: Path to WAV audio file
            language: Language code
            initial_prompt: Optional prompt
            
        Returns:
            List of WordTimestamp
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.transcribe(audio_path, language, initial_prompt)
        )
    
    @classmethod
    def preload_model(cls) -> None:
        """Preload the model during startup."""
        try:
            cls.get_model()
        except Exception as e:
            logger.error(f"Failed to preload Whisper model: {e}")
            raise
