"""
Transcription service using faster-whisper.
Supports multiple Vietnamese Whisper models with caching.
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Available Whisper models for Vietnamese
AVAILABLE_MODELS = {
    "EraX-WoW-Turbo": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
    "PhoWhisper Large": "kiendt/PhoWhisper-large-ct2"
}


@dataclass
class WordTimestamp:
    """A single word with precise timestamp."""
    word: str
    start: float
    end: float


class TranscriptionService:
    """
    Service for speech-to-text transcription using faster-whisper.
    Supports multiple models with caching.
    """
    
    _models: Dict[str, WhisperModel] = {}
    
    @classmethod
    def get_model(cls, model_name: str = "EraX-WoW-Turbo") -> WhisperModel:
        """
        Get or load a Whisper model (lazy loading with caching).
        
        Args:
            model_name: Name of the model from AVAILABLE_MODELS
            
        Returns:
            Loaded WhisperModel instance
        """
        cache_key = f"{model_name}_{settings.resolved_compute_type}"
        
        if cache_key in cls._models:
            return cls._models[cache_key]
        
        # Get model path
        if model_name in AVAILABLE_MODELS:
            model_path = AVAILABLE_MODELS[model_name]
        else:
            # Fallback to first available model
            model_name = list(AVAILABLE_MODELS.keys())[0]
            model_path = AVAILABLE_MODELS[model_name]
        
        logger.info(f"Loading Whisper model: {model_name} ({model_path})")
        logger.debug(f"Device: {settings.resolved_device}, Compute type: {settings.resolved_compute_type}")
        
        model = WhisperModel(
            model_path,
            device=settings.resolved_device,
            compute_type=settings.resolved_compute_type,
        )
        
        cls._models[cache_key] = model
        logger.info(f"✅ Whisper model loaded: {model_name}")
        
        return model
    
    @classmethod
    def is_loaded(cls, model_name: str = "EraX-WoW-Turbo") -> bool:
        """Check if a model is loaded."""
        cache_key = f"{model_name}_{settings.resolved_compute_type}"
        return cache_key in cls._models
    
    @classmethod
    def preload_model(cls, model_name: str = None) -> None:
        """Preload a model during startup."""
        if model_name is None:
            model_name = settings.default_whisper_model
        try:
            cls.get_model(model_name)
        except Exception as e:
            logger.error(f"Failed to preload Whisper model: {e}")
            raise
    
    @classmethod
    def transcribe_segment(
        cls,
        audio_array: np.ndarray,
        model_name: str = "EraX-WoW-Turbo",
        language: str = "vi",
        vad_options: Optional[dict] = None,
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe a numpy audio array segment.
        
        Args:
            audio_array: Audio data as numpy array (16kHz mono)
            model_name: Whisper model to use
            language: Language code (default: Vietnamese)
            vad_options: VAD filter options dict
            beam_size: Beam size for decoding
            temperature: Sampling temperature
            best_of: Number of candidates
            initial_prompt: Optional context prompt
            
        Returns:
            Transcribed text
        """
        model = cls.get_model(model_name)
        
        # Prepare VAD filter
        vad_filter = vad_options if vad_options else False
        
        # Process prompt
        prompt = initial_prompt.strip() if initial_prompt and initial_prompt.strip() else None
        
        # Run transcription
        segments_gen, info = model.transcribe(
            audio_array,
            language=language if language != "auto" else None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            temperature=temperature,
            best_of=best_of,
            initial_prompt=prompt,
            word_timestamps=False,
        )
        
        # Collect text
        text_parts = []
        for seg in segments_gen:
            text_parts.append(seg.text.strip())
        
        return " ".join(text_parts).strip()
    
    @classmethod
    async def transcribe_segment_async(
        cls,
        audio_array: np.ndarray,
        model_name: str = "EraX-WoW-Turbo",
        language: str = "vi",
        vad_options: Optional[dict] = None,
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """
        Async wrapper for transcription (runs in thread pool).
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.transcribe_segment(
                audio_array,
                model_name,
                language,
                vad_options,
                beam_size,
                temperature,
                best_of,
                initial_prompt
            )
        )
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Return list of available models."""
        return AVAILABLE_MODELS.copy()
