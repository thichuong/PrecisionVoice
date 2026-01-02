"""
Precision alignment service - Word-center-based speaker assignment.
Merges word-level transcription with speaker diarization using precise timestamps.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.core.config import get_settings
from app.schemas.models import TranscriptSegment
from app.services.transcription import WordTimestamp
from app.services.diarization import SpeakerSegment

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class WordWithSpeaker:
    """A word with assigned speaker."""
    word: str
    start: float
    end: float
    speaker: str


class AlignmentService:
    """
    Precision alignment service.
    Uses word-center-based algorithm for accurate speaker-to-text mapping.
    """
    
    # Pause threshold for splitting segments (seconds)
    PAUSE_THRESHOLD = 1.0
    
    @staticmethod
    def get_word_center(word: WordTimestamp) -> float:
        """Calculate the center time of a word."""
        return (word.start + word.end) / 2
    
    @classmethod
    def find_speaker_at_time(
        cls,
        time: float,
        speaker_segments: List[SpeakerSegment]
    ) -> Optional[str]:
        """
        Find which speaker is speaking at a given time.
        
        Args:
            time: Time point in seconds
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            Speaker label or None if no speaker found
        """
        for seg in speaker_segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        return None
    
    @classmethod
    def find_closest_speaker(
        cls,
        time: float,
        speaker_segments: List[SpeakerSegment]
    ) -> str:
        """
        Find the closest speaker to a given time (for gaps/silence).
        
        Args:
            time: Time point in seconds
            speaker_segments: List of speaker segments
            
        Returns:
            Closest speaker label or "Unknown"
        """
        if not speaker_segments:
            return "Unknown"
        
        min_distance = float('inf')
        closest_speaker = "Unknown"
        
        for seg in speaker_segments:
            # Distance to segment start or end
            dist_to_start = abs(time - seg.start)
            dist_to_end = abs(time - seg.end)
            min_seg_dist = min(dist_to_start, dist_to_end)
            
            if min_seg_dist < min_distance:
                min_distance = min_seg_dist
                closest_speaker = seg.speaker
        
        return closest_speaker
    
    @classmethod
    def assign_speakers_to_words(
        cls,
        words: List[WordTimestamp],
        speaker_segments: List[SpeakerSegment]
    ) -> List[WordWithSpeaker]:
        """
        Step 3c: Assign speakers to each word based on word center time.
        
        Args:
            words: List of words with timestamps from transcription
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            List of words with speaker assignments
        """
        if not speaker_segments:
            # No diarization available, assign all to "Speaker 1"
            logger.warning("No speaker segments available, using single speaker")
            return [
                WordWithSpeaker(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    speaker="Speaker 1"
                )
                for w in words
            ]
        
        words_with_speakers = []
        
        for word in words:
            # Calculate word center time
            center_time = cls.get_word_center(word)
            
            # Find speaker at this time
            speaker = cls.find_speaker_at_time(center_time, speaker_segments)
            
            # If no direct match, find closest speaker
            if speaker is None:
                speaker = cls.find_closest_speaker(center_time, speaker_segments)
            
            words_with_speakers.append(WordWithSpeaker(
                word=word.word,
                start=word.start,
                end=word.end,
                speaker=speaker
            ))
        
        logger.info(f"Assigned speakers to {len(words_with_speakers)} words")
        return words_with_speakers
    
    @classmethod
    def reconstruct_segments(
        cls,
        words_with_speakers: List[WordWithSpeaker]
    ) -> List[TranscriptSegment]:
        """
        Step 3d: Reconstruct sentence segments from words.
        
        Groups consecutive words of the same speaker into segments.
        Creates new segment when:
        - Speaker changes
        - Pause > PAUSE_THRESHOLD between words
        
        Args:
            words_with_speakers: List of words with speaker assignments
            
        Returns:
            List of TranscriptSegment with complete sentences
        """
        if not words_with_speakers:
            return []
        
        segments = []
        
        # Start first segment
        current_speaker = words_with_speakers[0].speaker
        current_start = words_with_speakers[0].start
        current_end = words_with_speakers[0].end
        current_words = [words_with_speakers[0].word]
        
        for i in range(1, len(words_with_speakers)):
            word = words_with_speakers[i]
            prev_word = words_with_speakers[i - 1]
            
            # Calculate pause between words
            pause = word.start - prev_word.end
            
            # Check if we need to start a new segment
            speaker_changed = word.speaker != current_speaker
            significant_pause = pause > cls.PAUSE_THRESHOLD
            
            if speaker_changed or significant_pause:
                # Save current segment
                segments.append(TranscriptSegment(
                    start=current_start,
                    end=current_end,
                    speaker=current_speaker,
                    text=" ".join(current_words)
                ))
                
                # Start new segment
                current_speaker = word.speaker
                current_start = word.start
                current_end = word.end
                current_words = [word.word]
            else:
                # Continue current segment
                current_end = word.end
                current_words.append(word.word)
        
        # Don't forget the last segment
        if current_words:
            segments.append(TranscriptSegment(
                start=current_start,
                end=current_end,
                speaker=current_speaker,
                text=" ".join(current_words)
            ))
        
        logger.info(f"Reconstructed {len(segments)} segments from {len(words_with_speakers)} words")
        return segments
    
    @classmethod
    def align_precision(
        cls,
        words: List[WordTimestamp],
        speaker_segments: List[SpeakerSegment]
    ) -> List[TranscriptSegment]:
        """
        Full precision alignment pipeline.
        
        Args:
            words: Word-level timestamps from transcription
            speaker_segments: Speaker segments from diarization
            
        Returns:
            List of TranscriptSegment with proper speaker assignments
        """
        # Step 3c: Assign speakers to words
        words_with_speakers = cls.assign_speakers_to_words(words, speaker_segments)
        
        # Step 3d: Reconstruct segments
        segments = cls.reconstruct_segments(words_with_speakers)
        
        return segments
    
    @staticmethod
    def format_timestamp_txt(seconds: float) -> str:
        """Format timestamp for TXT output: HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def format_timestamp_srt(seconds: float) -> str:
        """Format timestamp for SRT output: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @classmethod
    def generate_txt(cls, segments: List[TranscriptSegment], output_path: Path) -> Path:
        """
        Generate TXT transcript file.
        
        Format: [HH:MM:SS - HH:MM:SS] Speaker: Text
        """
        lines = []
        for seg in segments:
            start = cls.format_timestamp_txt(seg.start)
            end = cls.format_timestamp_txt(seg.end)
            lines.append(f"[{start} - {end}] {seg.speaker}: {seg.text}")
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated TXT: {output_path}")
        
        return output_path
    
    @classmethod
    def generate_srt(cls, segments: List[TranscriptSegment], output_path: Path) -> Path:
        """
        Generate SRT subtitle file.
        """
        lines = []
        for i, seg in enumerate(segments, 1):
            start = cls.format_timestamp_srt(seg.start)
            end = cls.format_timestamp_srt(seg.end)
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"[{seg.speaker}] {seg.text}")
            lines.append("")  # Empty line between entries
        
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated SRT: {output_path}")
        
        return output_path
    
    @classmethod
    def generate_outputs(
        cls,
        segments: List[TranscriptSegment],
        base_filename: str
    ) -> Tuple[Path, Path]:
        """Generate both TXT and SRT output files."""
        txt_path = settings.processed_dir / f"{base_filename}.txt"
        srt_path = settings.processed_dir / f"{base_filename}.srt"
        
        cls.generate_txt(segments, txt_path)
        cls.generate_srt(segments, srt_path)
        
        return txt_path, srt_path
