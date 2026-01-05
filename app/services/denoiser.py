"""
Speech Enhancement Service using SpeechBrain SepFormer.
Replaces Facebook Denoiser with state-of-the-art DNS4 model.
Provides noise reduction and dereverberation.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancementError(Exception):
    """Custom exception for speech enhancement errors."""
    pass


class EnhancementService:
    """
    Service for speech enhancement using SpeechBrain SepFormer DNS4.
    
    Features:
    - Advanced noise suppression (trained on Microsoft DNS4 Challenge)
    - Dereverberation
    - 16kHz output (perfect for Whisper)
    """
    
    _model = None
    _model_name: str = None
    
    @classmethod
    def _get_model(cls):
        """Lazy load the SpeechBrain SepFormer model."""
        if cls._model is None or cls._model_name != settings.enhancement_model:
            from speechbrain.inference.separation import SepformerSeparation
            
            logger.info(f"Loading SpeechBrain model: {settings.enhancement_model}")
            
            # Determine device
            device = settings.resolved_device
            run_opts = {"device": device} if device == "cuda" else {}
            
            cls._model = SepformerSeparation.from_hparams(
                source=settings.enhancement_model,
                savedir=str(settings.data_dir / "pretrained_models" / "sepformer-dns4"),
                run_opts=run_opts
            )
            
            cls._model_name = settings.enhancement_model
            logger.info(f"SpeechBrain model loaded on {device}")
            
        return cls._model

    @classmethod
    async def enhance_audio(cls, input_path: Path) -> Path:
        """
        Enhance audio by removing noise and reverb.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to enhanced WAV file
        """
        if not settings.enable_speech_enhancement:
            logger.debug("Speech enhancement disabled, skipping...")
            return input_path
            
        logger.debug(f"Starting speech enhancement for: {input_path.name}")
        
        try:
            # Run enhancement in executor to not block
            loop = asyncio.get_event_loop()
            enhanced_path = await loop.run_in_executor(
                None, 
                lambda: cls._run_enhancement(input_path)
            )
            
            logger.info(f"Speech enhancement complete: {enhanced_path.name}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Speech enhancement failed: {e}")
            # Fallback to original on failure rather than failing the whole pipeline
            logger.warning("Falling back to original audio.")
            return input_path

    @classmethod
    def _run_enhancement(cls, input_path: Path) -> Path:
        """Run the actual SpeechBrain enhancement (blocking)."""
        model = cls._get_model()
        
        # Load and check audio
        waveform, sample_rate = torchaudio.load(str(input_path))
        
        # Resample to 16kHz if needed (SepFormer DNS4 expects 16kHz)
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Save temporary file for SepFormer (it expects file path)
        temp_path = settings.processed_dir / f"{input_path.stem}_temp_for_enhance.wav"
        torchaudio.save(str(temp_path), waveform, sample_rate)
        
        try:
            # Run enhancement
            with torch.no_grad():
                est_sources = model.separate_file(path=str(temp_path))
            
            # est_sources shape: [batch, time, sources]
            # For enhancement, we take the first source (enhanced speech)
            enhanced = est_sources[:, :, 0].detach().cpu()
            
            # Ensure correct shape for torchaudio.save [channels, time]
            if enhanced.dim() == 1:
                enhanced = enhanced.unsqueeze(0)
            elif enhanced.dim() == 2 and enhanced.shape[0] > enhanced.shape[1]:
                # Shape is [time, channels], transpose it
                enhanced = enhanced.T
            
            # Save enhanced audio
            output_filename = f"{input_path.stem}_enhanced.wav"
            output_path = settings.processed_dir / output_filename
            
            torchaudio.save(str(output_path), enhanced, sample_rate)
            
            return output_path
            
        finally:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def preload_model(cls) -> None:
        """Preload the model during startup."""
        try:
            cls._get_model()
            logger.info("SpeechBrain enhancement model preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload enhancement model: {e}")
            # Don't raise - enhancement is optional


# Backward compatibility alias
DenoiserService = EnhancementService
