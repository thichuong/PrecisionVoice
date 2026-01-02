"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of the transcription process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    """A single segment of the transcript with speaker and timing."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker identifier")
    text: str = Field(..., description="Transcribed text")
    
    @property
    def start_formatted(self) -> str:
        """Format start time as HH:MM:SS."""
        return self._format_time(self.start)
    
    @property
    def end_formatted(self) -> str:
        """Format end time as HH:MM:SS."""
        return self._format_time(self.end)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TranscriptionRequest(BaseModel):
    """Request model for transcription settings."""
    language: str = Field(default="vi", description="Language code for transcription")
    num_speakers: Optional[int] = Field(default=None, description="Expected number of speakers (None for auto-detect)")
    output_format: str = Field(default="json", description="Output format: json, txt, srt")


class TranscriptionResponse(BaseModel):
    """Response containing the transcription results."""
    success: bool = Field(..., description="Whether transcription succeeded")
    message: str = Field(default="", description="Status message")
    segments: list[TranscriptSegment] = Field(default_factory=list, description="Transcript segments with speakers")
    duration: float = Field(default=0.0, description="Audio duration in seconds")
    num_speakers: int = Field(default=0, description="Number of detected speakers")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    download_txt: Optional[str] = Field(default=None, description="Download URL for TXT file")
    download_srt: Optional[str] = Field(default=None, description="Download URL for SRT file")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: bool = False
    device: str = "cpu"
