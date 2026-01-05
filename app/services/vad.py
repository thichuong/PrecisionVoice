"""
Voice Activity Detection (VAD) service using Silero VAD v5.
Detects speech segments and filters out silence to prevent Whisper hallucination.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torchaudio

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SpeechSegment:
    """A segment of detected speech with timestamps."""
    start: float  # seconds
    end: float    # seconds


class VADError(Exception):
    """Custom exception for VAD errors."""
    pass


class VADService:
    """
    Service for Voice Activity Detection using Silero VAD v5.
    Filters silence from audio to prevent Whisper hallucination.
    """
    
    _model = None
    _utils = None
    
    @classmethod
    def _get_model(cls):
        """Lazy load the Silero VAD model."""
        if cls._model is None:
            logger.debug("Loading Silero VAD model...")
            
            # Set number of threads for efficiency
            torch.set_num_threads(1)
            
            # Load model via silero-vad package
            from silero_vad import load_silero_vad
            cls._model = load_silero_vad()
            
            logger.debug("Silero VAD model loaded successfully")
        
        return cls._model
    
    @classmethod
    def detect_speech(
        cls,
        audio_path: Path,
        threshold: float = None,
        min_speech_duration_ms: int = None,
        min_silence_duration_ms: int = None,
        speech_pad_ms: int = None
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio file.
        
        Args:
            audio_path: Path to WAV audio file (16kHz mono)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration to keep
            min_silence_duration_ms: Minimum silence to split segments
            speech_pad_ms: Padding around speech segments
            
        Returns:
            List of SpeechSegment with start/end times in seconds
        """
        from silero_vad import read_audio, get_speech_timestamps
        
        model = cls._get_model()
        
        # Use settings defaults if not provided
        threshold = threshold or settings.silero_vad_threshold
        min_speech_duration_ms = min_speech_duration_ms or settings.silero_vad_min_speech_ms
        min_silence_duration_ms = min_silence_duration_ms or settings.silero_vad_min_silence_ms
        speech_pad_ms = speech_pad_ms or settings.silero_vad_speech_pad_ms
        
        logger.debug(f"Running VAD on: {audio_path.name}")
        logger.debug(f"VAD params: threshold={threshold}, min_speech={min_speech_duration_ms}ms, min_silence={min_silence_duration_ms}ms")
        
        # Read audio (silero_vad handles resampling to 16kHz if needed)
        wav = read_audio(str(audio_path))
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True  # Return in seconds for easier handling
        )
        
        # Convert to SpeechSegment objects
        segments = [
            SpeechSegment(start=ts['start'], end=ts['end'])
            for ts in speech_timestamps
        ]
        
        logger.info(f"VAD detected {len(segments)} speech segments in {audio_path.name}")
        
        return segments
    
    @classmethod
    def extract_speech_audio(
        cls,
        audio_path: Path,
        speech_segments: List[SpeechSegment],
        output_path: Path = None
    ) -> Tuple[Path, List[SpeechSegment]]:
        """
        Extract and concatenate speech segments into a new audio file.
        Returns the new audio path and adjusted segment timestamps.
        
        Args:
            audio_path: Path to original audio file
            speech_segments: List of detected speech segments
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Tuple of (output_path, adjusted_segments)
            - adjusted_segments have timestamps relative to the new audio
        """
        if not speech_segments:
            logger.warning("No speech segments provided, returning original audio")
            return audio_path, []
        
        # Load original audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Extract speech segments
        extracted_segments = []
        adjusted_segments = []
        current_position = 0.0
        
        for segment in speech_segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            
            # Clamp to valid range
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            
            if end_sample > start_sample:
                segment_audio = waveform[:, start_sample:end_sample]
                extracted_segments.append(segment_audio)
                
                # Calculate duration
                duration = (end_sample - start_sample) / sample_rate
                
                # Track adjusted timestamps (relative to concatenated audio)
                adjusted_segments.append(SpeechSegment(
                    start=current_position,
                    end=current_position + duration
                ))
                current_position += duration
        
        if not extracted_segments:
            logger.warning("No valid audio extracted, returning original")
            return audio_path, []
        
        # Concatenate all speech segments
        concatenated = torch.cat(extracted_segments, dim=1)
        
        # Generate output path
        if output_path is None:
            output_path = settings.processed_dir / f"{audio_path.stem}_vad.wav"
        
        # Save concatenated audio
        torchaudio.save(str(output_path), concatenated, sample_rate)
        
        total_original = waveform.shape[1] / sample_rate
        total_filtered = concatenated.shape[1] / sample_rate
        reduction = (1 - total_filtered / total_original) * 100
        
        logger.info(f"VAD filtering complete: {total_original:.1f}s -> {total_filtered:.1f}s ({reduction:.1f}% reduction)")
        
        return output_path, adjusted_segments
    
    @classmethod
    async def filter_silence(
        cls,
        input_path: Path
    ) -> Tuple[Path, List[SpeechSegment], List[SpeechSegment]]:
        """
        Async wrapper to detect and filter silence from audio.
        
        Args:
            input_path: Path to WAV audio file
            
        Returns:
            Tuple of (filtered_audio_path, original_segments, adjusted_segments)
            - original_segments: timestamps relative to original audio
            - adjusted_segments: timestamps relative to filtered audio
        """
        if not settings.enable_silero_vad:
            logger.debug("Silero VAD disabled, skipping...")
            return input_path, [], []
        
        logger.debug(f"Starting VAD filtering for: {input_path.name}")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Detect speech segments
            original_segments = await loop.run_in_executor(
                None,
                lambda: cls.detect_speech(input_path)
            )
            
            if not original_segments:
                logger.warning("No speech detected in audio, returning original")
                return input_path, [], []
            
            # Extract and concatenate speech
            filtered_path, adjusted_segments = await loop.run_in_executor(
                None,
                lambda: cls.extract_speech_audio(input_path, original_segments)
            )
            
            return filtered_path, original_segments, adjusted_segments
            
        except Exception as e:
            logger.error(f"VAD filtering failed: {e}")
            logger.warning("Falling back to original audio")
            return input_path, [], []
    
    @classmethod
    def map_timestamps_to_original(
        cls,
        word_timestamps: list,
        original_segments: List[SpeechSegment],
        adjusted_segments: List[SpeechSegment]
    ) -> list:
        """
        Map word timestamps from filtered audio back to original audio timeline.
        
        Args:
            word_timestamps: List of WordTimestamp from Whisper
            original_segments: Speech segments in original audio
            adjusted_segments: Corresponding segments in filtered audio
            
        Returns:
            List of WordTimestamp with corrected timestamps
        """
        if not original_segments or not adjusted_segments:
            return word_timestamps
        
        if len(original_segments) != len(adjusted_segments):
            logger.warning("Segment count mismatch, returning original timestamps")
            return word_timestamps
        
        # Build mapping: for each position in filtered audio,
        # calculate the corresponding position in original audio
        corrected = []
        
        for word in word_timestamps:
            # Find which segment this word belongs to
            new_start = cls._map_time(word.start, original_segments, adjusted_segments)
            new_end = cls._map_time(word.end, original_segments, adjusted_segments)
            
            # Create new word with corrected timestamps
            from app.services.transcription import WordTimestamp
            corrected.append(WordTimestamp(
                word=word.word,
                start=new_start,
                end=new_end
            ))
        
        return corrected
    
    @classmethod
    def _map_time(
        cls,
        filtered_time: float,
        original_segments: List[SpeechSegment],
        adjusted_segments: List[SpeechSegment]
    ) -> float:
        """Map a time from filtered audio to original audio."""
        for orig, adj in zip(original_segments, adjusted_segments):
            if adj.start <= filtered_time <= adj.end:
                # Calculate offset within segment
                offset = filtered_time - adj.start
                return orig.start + offset
        
        # If not found, try to extrapolate from last segment
        if filtered_time > adjusted_segments[-1].end:
            last_orig = original_segments[-1]
            last_adj = adjusted_segments[-1]
            overflow = filtered_time - last_adj.end
            return last_orig.end + overflow
        
        # Default: return as-is
        return filtered_time
    
    @classmethod
    def preload_model(cls) -> None:
        """Preload the VAD model during startup."""
        try:
            cls._get_model()
            logger.info("Silero VAD model preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload VAD model: {e}")
