---
title: PrecisionVoice
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app/main.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# PrecisionVoice - STT & Speaker Diarization

A production-ready Speech-to-Text and Speaker Diarization web application using FastAPI, faster-whisper, and pyannote.audio.

## Features

- 🎙️ Speech-to-Text using `kiendt/PhoWhisper-large-ct2` (optimized for Vietnamese)
- 👥 Speaker Diarization using `pyannote/speaker-diarization-3.1`
- 🧼 Advanced Denoising using Facebook's `Denoiser` (dns64)
- 🎤 Vocal Isolation using `MDX-Net` (UVR-MDX-NET-Voc_FT)
- 🔄 Automatic speaker-transcript alignment
- 📥 Download results in TXT or SRT format
- 🐳 Docker-ready with persistent model caching and GPU support

## Quick Start

### Prerequisites

1. Docker and Docker Compose
2. (Optional) NVIDIA GPU with CUDA support
3. HuggingFace account with access to pyannote models

### Setup

1. Clone and configure:
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

2. Build and run:
   ```bash
   docker compose up --build
   ```

3. Open http://localhost:8000

## Audio Processing Pipeline

The system uses a state-of-the-art multi-stage pipeline to ensure maximum accuracy:

1. **Speech Enhancement**: Background noise, hums, and interference are removed using Facebook's `Denoiser` (Deep Learning Wave-U-Net).
2. **Vocal Isolation**: Vocals are stripped from any remaining background music or non-speech sounds using `MDX-Net`.
3. **Refinement**: Subtle highpass filtering and EBU R128 loudness normalization for consistent volume.
4. **Transcription**: High-precision Vietnamese transcription using `PhoWhisper`.
5. **Diarization**: Segmenting audio by speaker.
6. **Alignment**: Merging transcripts with speaker segments.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | Required for Pyannote models |
| `ENABLE_DENOISER` | `True` | Toggle Facebook speech enhancement |
| `DENOISER_MODEL` | `dns64` | Model for denoising |
| `ENABLE_VOCAL_SEPARATION` | `True` | Toggle MDX-Net vocal isolation |
| `MDX_MODEL` | `UVR-MDX-NET-Voc_FT` | Model for vocal separation |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

## Development

### Local Setup (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/transcribe` | POST | Upload and transcribe audio |
| `/api/download/{filename}` | GET | Download result files |

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG

## License

MIT
