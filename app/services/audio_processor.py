"""
Audio processing service using FFmpeg.
Handles file validation, conversion to 16kHz mono WAV, and cleanup.
"""
import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

import ffmpeg

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


class AudioProcessor:
    """Service for processing audio files."""
    
    ALLOWED_EXTENSIONS = settings.allowed_extensions
    TARGET_SAMPLE_RATE = settings.sample_rate
    TARGET_CHANNELS = settings.channels
    
    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> bool:
        """
        Validate uploaded file.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            
        Returns:
            True if valid
            
        Raises:
            AudioProcessingError: If validation fails
        """
        # Check extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise AudioProcessingError(
                f"Invalid file type: .{ext}. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )
        
        # Check size
        if file_size > settings.max_upload_size_bytes:
            raise AudioProcessingError(
                f"File too large: {file_size / (1024*1024):.1f}MB. "
                f"Maximum: {settings.max_upload_size_mb}MB"
            )
        
        return True
    
    @classmethod
    async def save_upload(cls, file_content: bytes, original_filename: str) -> Path:
        """
        Save uploaded file to temporary location.
        
        Args:
            file_content: File bytes
            original_filename: Original filename for extension
            
        Returns:
            Path to saved file
        """
        ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else 'wav'
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}.{ext}"
        filepath = settings.upload_dir / filename
        
        # Write file asynchronously
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: filepath.write_bytes(file_content))
        
        logger.info(f"Saved upload: {filepath}")
        return filepath
    
    @classmethod
    async def convert_to_wav(cls, input_path: Path) -> Path:
        """
        Convert audio to 16kHz mono WAV using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to converted WAV file
        """
        output_filename = f"{input_path.stem}_processed.wav"
        output_path = settings.processed_dir / output_filename
        
        try:
            # Run ffmpeg conversion in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: cls._run_ffmpeg_conversion(input_path, output_path))
            
            logger.info(f"Converted to WAV: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise AudioProcessingError(f"Audio conversion failed: {error_msg}")
    
    @staticmethod
    def _run_ffmpeg_conversion(input_path: Path, output_path: Path) -> None:
        """Run the actual FFmpeg conversion (blocking)."""
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',
                ar=16000,
                ac=1
            )
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
    
    @classmethod
    async def get_audio_duration(cls, filepath: Path) -> float:
        """
        Get audio file duration in seconds.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            loop = asyncio.get_event_loop()
            probe = await loop.run_in_executor(
                None, 
                lambda: ffmpeg.probe(str(filepath))
            )
            
            duration = float(probe['format'].get('duration', 0))
            return duration
            
        except ffmpeg.Error as e:
            logger.warning(f"Could not probe audio duration: {e}")
            return 0.0
    
    @classmethod
    async def cleanup_files(cls, *filepaths: Path) -> None:
        """
        Delete temporary files.
        
        Args:
            filepaths: Paths to files to delete
        """
        for filepath in filepaths:
            try:
                if filepath and filepath.exists():
                    filepath.unlink()
                    logger.info(f"Cleaned up: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to clean up {filepath}: {e}")
    
    @classmethod
    async def process_upload(cls, file_content: bytes, filename: str) -> Tuple[Path, float]:
        """
        Full upload processing pipeline: validate, save, convert.
        
        Args:
            file_content: Uploaded file bytes
            filename: Original filename
            
        Returns:
            Tuple of (processed WAV path, duration in seconds)
        """
        # Validate
        cls.validate_file(filename, len(file_content))
        
        # Save original
        original_path = await cls.save_upload(file_content, filename)
        
        try:
            # Convert to WAV
            wav_path = await cls.convert_to_wav(original_path)
            
            # Get duration
            duration = await cls.get_audio_duration(wav_path)
            
            # Cleanup original (keep WAV for processing)
            await cls.cleanup_files(original_path)
            
            return wav_path, duration
            
        except Exception as e:
            # Cleanup on error
            await cls.cleanup_files(original_path)
            raise
