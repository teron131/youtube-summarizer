# YouTube Summarizer Backend API

A Python backend API for YouTube video analysis with AI-powered transcription and summarization capabilities. Built with FastAPI for high-performance video processing.

## 🚀 Features

- **Video Processing**: Download and process audio from YouTube videos
- **AI Transcription**: Automatic speech-to-text using Fal.ai
- **Smart Summarization**: AI-powered content summarization using Gemini
- **RESTful API**: Clean FastAPI endpoints for programmatic access
- **Robust Error Handling**: Comprehensive logging and error reporting
- **Caption Detection**: Uses existing captions when available to save processing time

## 🏗️ Architecture

```
youtube-summarizer/
├── app.py                    # FastAPI application server
├── youtube_summarizer/       # Core processing modules
│   ├── __init__.py
│   ├── youtube_loader.py     # YouTube data extraction
│   ├── transcriber.py        # Audio transcription
│   ├── summarizer.py         # AI summarization
│   └── utils.py             # Utility functions
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── start.sh                # Production startup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- FFmpeg (for audio processing)
- API Keys: `FAL_KEY`, `GEMINI_API_KEY`

### 1. Installation

```bash
git clone <repository-url>
cd youtube-summarizer

# Install with UV (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Install as editable package
uv pip install -e .
# Or: pip install -e .
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env_example .env

# Edit .env with your API keys
```

### 3. Start the API Server

```bash
# Development mode
python app.py

# Or using uvicorn directly
python -m uvicorn app:app --host 0.0.0.0 --port 8080

# Production mode
./start.sh
```

### 4. Access the API

- **API Server**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/api/health

## 🎯 API Endpoints

### POST /api/process
Process a YouTube video with transcription and summarization.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "generate_summary": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Video processed successfully",
  "data": {
    "title": "Video Title",
    "author": "Channel Name", 
    "transcript": "Full transcription text...",
    "summary": "AI-generated summary...",
    "processing_time": "45.2s",
    "url": "original_url"
  },
  "logs": ["Processing step logs..."]
}
```

### POST /api/video-info
Extract basic video information without processing.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:**
```json
{
  "title": "Video Title",
  "author": "Channel Name",
  "duration": "12:34",
  "thumbnail": "thumbnail_url",
  "view_count": 123456,
  "upload_date": "20240101"
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "YouTube Summarizer API is running",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🛠️ Development

### Package Usage

You can also use this as a Python package:

```python
from youtube_summarizer.youtube_loader import extract_video_info, download_audio_bytes
from youtube_summarizer.transcriber import transcribe_with_fal
from youtube_summarizer.summarizer import quick_summary

# Extract video info
info = extract_video_info("https://www.youtube.com/watch?v=VIDEO_ID")

# Download and transcribe
audio_bytes = download_audio_bytes(info)
transcript = transcribe_with_fal(audio_bytes)

# Generate summary
summary = quick_summary(transcript)
```

### Running Tests

```bash
# Install development dependencies
uv add --dev pytest black ruff

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .
```

## 🚦 Processing Pipeline

1. **Video Info Extraction**: Get metadata from YouTube URL
2. **Caption Check**: Look for existing captions first
3. **Audio Download**: Extract audio if no captions found  
4. **Audio Optimization**: Prepare for transcription
5. **AI Transcription**: Convert speech to text via Fal.ai
6. **Text Formatting**: Clean and structure transcription
7. **AI Summarization**: Generate summary via Gemini
8. **Response**: Return structured data

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FAL_KEY` | ✅ | Fal.ai API key for transcription | - |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for summarization | - |
| `PORT` | ❌ | Server port | 8080 |
| `HOST` | ❌ | Server host | 0.0.0.0 |

### Dependencies

Core dependencies:
- **FastAPI**: Web framework
- **yt-dlp**: YouTube video processing
- **pydub**: Audio manipulation
- **google-genai**: Gemini AI integration
- **fal-client**: Transcription service
- **uvicorn**: ASGI server

## 🔍 Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian  
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

**API Key errors:**
- Ensure `FAL_KEY` is set for transcription
- Ensure `GEMINI_API_KEY` is set for summarization
- Check `.env` file is properly configured

**Import errors:**
```bash
# Install package in editable mode
uv pip install -e .
```

## 📚 Project Structure

### Core Modules

- **`youtube_loader.py`**: YouTube data extraction and audio downloading
- **`transcriber.py`**: Audio transcription using Fal.ai
- **`summarizer.py`**: AI summarization using Gemini
- **`utils.py`**: Utility functions and helpers

### API Structure

- **`app.py`**: Main FastAPI application
- **Request/Response Models**: Pydantic models for API validation
- **Error Handling**: Comprehensive error reporting and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🔗 Quick Reference

- **API Server**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/api/health
- **OpenAPI Schema**: http://localhost:8080/openapi.json