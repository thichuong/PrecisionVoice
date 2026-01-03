# PrecisionVoice - STT & Speaker Diarization

A production-ready Speech-to-Text and Speaker Diarization web application using FastAPI, faster-whisper, and pyannote.audio.

## Features

- 🎙️ Speech-to-Text using `kiendt/PhoWhisper-large-ct2` (optimized for Vietnamese)
- 👥 Speaker Diarization using `pyannote/speaker-diarization-3.1`
- 🔄 Automatic speaker-transcript alignment
- 📥 Download results in TXT or SRT format
- 🐳 Docker-ready with GPU support

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
