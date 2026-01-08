"""
Audio processing utilities.
Simple validation and file handling.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


class AudioProcessor:
    """
    Simple audio processor for file validation and handling.
    Audio conversion is done in processor.py using ffmpeg.
    """
    
    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> None:
        """
        Validate uploaded file.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            
        Raises:
            AudioProcessingError: If validation fails
        """
        # Check extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in settings.allowed_extensions:
            raise AudioProcessingError(
                f"File type '.{ext}' not supported. "
                f"Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Check size
        if file_size > settings.max_upload_size_bytes:
            raise AudioProcessingError(
                f"File too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size: {settings.max_upload_size_mb}MB"
            )
    
    @classmethod
    async def save_upload(cls, file_content: bytes, original_filename: str) -> Path:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: Raw file bytes
            original_filename: Original filename for extension
            
        Returns:
            Path to saved file
        """
        import aiofiles
        
        # Generate unique filename
        ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else 'wav'
        unique_filename = f"{uuid.uuid4()}.{ext}"
        file_path = settings.upload_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"Saved upload: {file_path} ({len(file_content) / 1024:.1f}KB)")
        return file_path
    
    @classmethod
    async def cleanup_files(cls, *paths: Path) -> None:
        """Remove temporary files."""
        import asyncio
        
        for path in paths:
            try:
                if path and path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")
