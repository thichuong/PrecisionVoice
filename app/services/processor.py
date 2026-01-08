"""
Main audio processor following notebook workflow.
Diarize-First approach: Diarize -> Slice -> Transcribe each segment.
"""
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import librosa

from app.core.config import get_settings
from app.services.transcription import TranscriptionService
from app.services.diarization import DiarizationService, SpeakerSegment

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TranscriptSegment:
    """A transcribed segment with speaker info."""
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    segments: List[TranscriptSegment]
    speaker_count: int
    duration: float
    processing_time: float
    
    # Output files
    txt_content: str = ""
    srt_content: str = ""


def convert_audio_to_wav(audio_path: Path) -> Path:
    """
    Convert any audio to WAV 16kHz Mono using ffmpeg.
    
    Args:
        audio_path: Input audio file path
        
    Returns:
        Path to converted WAV file
    """
    output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
    
    # Remove existing file
    if output_path.exists():
        output_path.unlink()
    
    command = [
        "ffmpeg",
        "-i", str(audio_path),
        "-ar", "16000",  # Sample rate 16kHz
        "-ac", "1",       # Mono channel
        "-y",             # Overwrite output
        str(output_path)
    ]
    
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Converted audio to WAV: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        # Fallback: return original if conversion fails
        return audio_path


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS.ms or HH:MM:SS.ms"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def merge_consecutive_segments(
    segments: List[SpeakerSegment], 
    max_gap: float = 0.5
) -> List[SpeakerSegment]:
    """Merge consecutive segments from same speaker."""
    if not segments:
        return []
    
    merged = []
    current = SpeakerSegment(
        start=segments[0].start,
        end=segments[0].end,
        speaker=segments[0].speaker
    )
    
    for seg in segments[1:]:
        if seg.speaker == current.speaker and (seg.start - current.end) <= max_gap:
            # Merge: extend current segment
            current.end = seg.end
        else:
            # New speaker or gap too large
            merged.append(current)
            current = SpeakerSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker
            )
    
    merged.append(current)
    return merged


class Processor:
    """
    Main processor following notebook workflow.
    
    Workflow:
    1. Convert audio to WAV 16kHz mono
    2. Load audio with librosa
    3. Run diarization (pyannote)
    4. Merge consecutive segments if requested
    5. Slice audio and transcribe each segment
    6. Generate output
    """
    
    @classmethod
    async def process_audio(
        cls,
        audio_path: Path,
        model_name: str = "EraX-WoW-Turbo",
        language: str = "vi",
        merge_segments: bool = True,
        # VAD options
        vad_filter: bool = True,
        vad_min_silence_ms: int = 1000,
        vad_speech_pad_ms: int = 400,
        vad_min_speech_ms: int = 250,
        vad_threshold: float = 0.5,
        # Generation options
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Process audio file with diarization and transcription.
        
        Args:
            audio_path: Path to input audio file
            model_name: Whisper model to use
            language: Language code
            merge_segments: Whether to merge consecutive speaker segments
            ... various Whisper parameters
            
        Returns:
            ProcessingResult with transcribed segments and output files
        """
        import asyncio
        
        total_start = time.time()
        
        # Step 1: Convert to WAV
        logger.info("Step 1: Converting audio to WAV 16kHz...")
        wav_path = await asyncio.get_event_loop().run_in_executor(
            None, convert_audio_to_wav, audio_path
        )
        
        # Step 2: Load audio with librosa
        logger.info("Step 2: Loading audio...")
        y, sr = await asyncio.get_event_loop().run_in_executor(
            None, lambda: librosa.load(str(wav_path), sr=16000)
        )
        duration = len(y) / sr
        logger.info(f"Audio loaded: {duration:.1f}s, {sr}Hz")
        
        # Step 3: Diarization
        logger.info("Step 3: Running diarization...")
        try:
            diarization_segments = await DiarizationService.diarize_async(wav_path)
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Fallback: create single segment for whole audio
            diarization_segments = [SpeakerSegment(
                start=0.0,
                end=duration,
                speaker="Speaker 1"
            )]
        
        # Sort by start time
        diarization_segments.sort(key=lambda x: x.start)
        
        # Step 4: Merge if requested
        if merge_segments and diarization_segments:
            logger.info("Step 4: Merging consecutive segments...")
            diarization_segments = merge_consecutive_segments(diarization_segments)
        
        # Step 5: Transcribe each segment
        logger.info(f"Step 5: Transcribing {len(diarization_segments)} segments...")
        
        # Prepare VAD options
        vad_options = None
        if vad_filter:
            vad_options = {
                "min_silence_duration_ms": vad_min_silence_ms,
                "speech_pad_ms": vad_speech_pad_ms,
                "min_speech_duration_ms": vad_min_speech_ms,
                "threshold": vad_threshold
            }
        
        processed_segments: List[TranscriptSegment] = []
        unique_speakers = set()
        
        for idx, seg in enumerate(diarization_segments):
            logger.info(f"Transcribing segment {idx+1}/{len(diarization_segments)} ({seg.speaker})...")
            
            # Slice audio
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            
            if end_sample <= start_sample:
                continue
            
            y_seg = y[start_sample:end_sample]
            
            # Transcribe
            try:
                text = await TranscriptionService.transcribe_segment_async(
                    audio_array=y_seg,
                    model_name=model_name,
                    language=language,
                    vad_options=vad_options,
                    beam_size=beam_size,
                    temperature=temperature,
                    best_of=best_of,
                    initial_prompt=initial_prompt
                )
                
                if text.strip():
                    unique_speakers.add(seg.speaker)
                    processed_segments.append(TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker,
                        text=text.strip()
                    ))
            except Exception as e:
                logger.error(f"Error transcribing segment {idx}: {e}")
                continue
        
        processing_time = time.time() - total_start
        logger.info(f"Processing complete: {len(processed_segments)} segments, {len(unique_speakers)} speakers in {processing_time:.1f}s")
        
        # Step 6: Generate outputs
        txt_content = cls._generate_txt(processed_segments, unique_speakers, processing_time, duration)
        srt_content = cls._generate_srt(processed_segments)
        
        # Cleanup WAV if different from original
        if wav_path != audio_path and wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass
        
        return ProcessingResult(
            segments=processed_segments,
            speaker_count=len(unique_speakers),
            duration=duration,
            processing_time=processing_time,
            txt_content=txt_content,
            srt_content=srt_content
        )
    
    @classmethod
    def _generate_txt(
        cls,
        segments: List[TranscriptSegment],
        speakers: set,
        processing_time: float,
        duration: float
    ) -> str:
        """Generate plain text transcript."""
        lines = [
            "# Transcription Result",
            f"# Duration: {format_timestamp(duration)}",
            f"# Speakers: {len(speakers)}",
            f"# Processing time: {processing_time:.1f}s",
            "",
        ]
        
        speaker_icons = {
            'Speaker 1': '🔵',
            'Speaker 2': '🟢',
            'Speaker 3': '🟡',
            'Speaker 4': '🟠',
            'Speaker 5': '🔴',
            'Speaker 6': '🟣',
        }
        
        for seg in segments:
            ts = f"[{format_timestamp(seg.start)} → {format_timestamp(seg.end)}]"
            icon = speaker_icons.get(seg.speaker, '⚪')
            lines.append(f"{ts} {icon} {seg.speaker}: {seg.text}")
        
        return "\n".join(lines)
    
    @classmethod
    def _generate_srt(cls, segments: List[TranscriptSegment]) -> str:
        """Generate SRT subtitle format."""
        lines = []
        
        for idx, seg in enumerate(segments, 1):
            start_ts = format_srt_timestamp(seg.start)
            end_ts = format_srt_timestamp(seg.end)
            
            lines.append(str(idx))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"[{seg.speaker}] {seg.text}")
            lines.append("")
        
        return "\n".join(lines)
