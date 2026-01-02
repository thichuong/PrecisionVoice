"""
Application configuration using Pydantic Settings.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # HuggingFace
    hf_token: str = ""
    enable_noise_reduction: bool = True
    
    # Model settings
    whisper_model: str = "kiendt/PhoWhisper-large-ct2"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    
    # Device settings
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "float16"  # float16 for GPU, int8 for CPU
    
    # Upload settings
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
    
    # Audio processing settings
    sample_rate: int = 16000
    channels: int = 1  # Mono
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    upload_dir: Path = data_dir / "uploads"
    processed_dir: Path = data_dir / "processed"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
    
    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device
    
    @property
    def resolved_compute_type(self) -> str:
        """Get appropriate compute type for device."""
        if self.resolved_device == "cuda":
            return "float16"
        return "int8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
