"""
Vocal Separation Service using Demucs (Hybrid Transformer).
Isolates vocals from audio files to improve speech recognition accuracy.
"""
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VocalSeparationError(Exception):
    """Custom exception for vocal separation errors."""
    pass


class VocalSeparator:
    """
    Service for separating vocals from audio using Demucs.
    
    Uses the Hybrid Transformer Demucs model (htdemucs) which produces
    4 stems: drums, bass, other, vocals. Only the vocals stem is kept.
    """
    
    _model = None
    _model_name: str = None
    
    @classmethod
    def _get_model(cls):
        """Lazy load the Demucs model."""
        if cls._model is None or cls._model_name != settings.demucs_model:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
            
            logger.info(f"Loading Demucs model: {settings.demucs_model}")
            model = get_model(settings.demucs_model)
            
            # Wrap in BagOfModels if needed
            if not isinstance(model, BagOfModels):
                model = BagOfModels([model])
            
            # Move to appropriate device
            device = settings.resolved_device
            model.to(device)
            model.eval()
            
            cls._model = model
            cls._model_name = settings.demucs_model
            logger.info(f"Demucs model loaded on {device}")
        
        return cls._model
    
    @classmethod
    async def separate_vocals(cls, input_path: Path) -> Path:
        """
        Separate vocals from audio file using Demucs.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to separated vocals WAV file
        """
        if not settings.enable_vocal_separation:
            logger.info("Vocal separation disabled, skipping...")
            return input_path
        
        logger.info(f"Starting vocal separation for: {input_path.name}")
        
        try:
            # Run separation in executor to not block
            loop = asyncio.get_event_loop()
            vocals_path = await loop.run_in_executor(
                None, 
                lambda: cls._run_separation(input_path)
            )
            
            logger.info(f"Vocal separation complete: {vocals_path.name}")
            return vocals_path
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {e}")
            raise VocalSeparationError(f"Vocal separation failed: {e}")
    
    @classmethod
    def _run_separation(cls, input_path: Path) -> Path:
        """Run the actual Demucs separation (blocking)."""
        import torchaudio
        from demucs.audio import AudioFile
        from demucs.apply import apply_model
        
        model = cls._get_model()
        device = settings.resolved_device
        
        # Load audio
        logger.info("Loading audio for separation...")
        audio_file = AudioFile(input_path)
        wav = audio_file.read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels
        )
        
        # Move to device
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.to(device)
        
        # Apply model
        logger.info("Running Demucs model...")
        with torch.no_grad():
            sources = apply_model(
                model, 
                wav[None],
                device=device,
                progress=False,
                num_workers=0
            )[0]
        
        # Sources order: drums, bass, other, vocals
        # We want index 3 (vocals)
        sources = sources * ref.std() + ref.mean()
        vocals = sources[3]  # Get vocals stem
        
        # Save vocals to file
        output_filename = f"{input_path.stem}_vocals.wav"
        output_path = settings.processed_dir / output_filename
        
        logger.info(f"Saving vocals to: {output_path}")
        torchaudio.save(
            str(output_path),
            vocals.cpu(),
            model.samplerate
        )
        
        return output_path
