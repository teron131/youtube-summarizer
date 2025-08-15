"""
YouTube Summarizer FastAPI Application
--------------------------------------

This FastAPI application provides an API to download, transcribe, and summarize
YouTube videos. It serves a simple web interface and exposes a processing endpoint.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from youtube_summarizer.summarizer import quick_summary, simple_format_subtitle
from youtube_summarizer.transcriber import (
    optimize_audio_for_transcription,
    transcribe_with_fal,
)
from youtube_summarizer.utils import log_and_print
from youtube_summarizer.youtube_loader import (
    download_audio_bytes,
    extract_video_info,
    get_subtitle_from_captions,
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="YouTube Summarizer",
    description="Standalone YouTube processing with transcription & summarization",
)

# CORS configuration: allow multiple origins or all via env var
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class YouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


# API Routes
@app.get("/", response_class=FileResponse)
async def get_web_interface():
    """Serve the web interface."""
    return FileResponse("frontend.html")


@app.post("/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeRequest):
    """
    Process YouTube video with robust error handling and visible logging.
    This endpoint orchestrates the entire workflow from video info extraction
    to final summary generation.
    """
    start_time = datetime.now()
    logs = [f"🎬 Starting processing: {request.url}"]

    try:
        log_and_print("📋 Step 1: Extracting video info...")
        logs.append("📋 Step 1: Extracting video info...")
        info = extract_video_info(request.url)
        title = info.get("title", "Unknown")
        author = info.get("uploader", "Unknown")
        log_and_print(f"✅ Video found: {title} by {author}")
        logs.append(f"✅ Video found: {title} by {author}")

        log_and_print("📋 Step 2: Checking for existing captions...")
        logs.append("📋 Step 2: Checking for existing captions...")
        subtitle = get_subtitle_from_captions(info)

        if subtitle:
            log_and_print("✅ Found existing captions - skipping transcription")
            logs.append("✅ Found existing captions - skipping transcription")
            formatted_subtitle = simple_format_subtitle(subtitle)
        else:
            log_and_print("🎯 No captions found")
            logs.append("🎯 No captions found")
            try:
                log_and_print("📋 Step 3: Downloading audio...")
                logs.append("📋 Step 3: Downloading audio...")
                audio_bytes = download_audio_bytes(info)

                log_and_print("📋 Step 4: Optimizing and transcribing audio...")
                logs.append("📋 Step 4: Optimizing and transcribing audio...")
                optimized_audio = optimize_audio_for_transcription(audio_bytes)

                if not os.getenv("FAL_KEY"):
                    log_and_print("❌ FAL_KEY not configured")
                    logs.append("❌ FAL_KEY not configured")
                    formatted_subtitle = "[FAL_KEY not configured]"
                else:
                    subtitle = transcribe_with_fal(optimized_audio)
                    formatted_subtitle = simple_format_subtitle(subtitle)
                    log_and_print("✅ Transcription completed")
                    logs.append("✅ Transcription completed")
            except Exception as audio_error:
                error_msg = f"❌ Audio processing failed: {str(audio_error)}"
                log_and_print(error_msg)
                logs.append(error_msg)
                formatted_subtitle = f"[Audio processing failed: {str(audio_error)}]"

        summary = None
        if request.generate_summary and not formatted_subtitle.startswith("["):
            log_and_print("📋 Step 5: Generating summary...")
            logs.append("📋 Step 5: Generating summary...")
            full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{formatted_subtitle}"
            summary = quick_summary(full_content)
            log_and_print("✅ Summary generated")
            logs.append("✅ Summary generated")

        processing_time = datetime.now() - start_time
        completion_msg = f"✅ Processing completed in {processing_time.total_seconds():.1f}s"
        log_and_print(completion_msg)
        logs.append(completion_msg)

        result_data = {
            "title": title,
            "author": author,
            "subtitle": formatted_subtitle,
            "summary": summary,
            "processing_time": f"{processing_time.total_seconds():.1f}s",
            "url": request.url,
        }

        return ProcessingResponse(
            status="success",
            message="Video processed successfully",
            data=result_data,
            logs=logs,
        )

    except Exception as e:
        processing_time = datetime.now() - start_time
        error_message = f"Processing error: {str(e)}"
        failure_msg = f"💔 Failed after {processing_time.total_seconds():.1f}s"
        log_and_print(f"❌ {error_message}")
        log_and_print(failure_msg)
        logs.append(f"❌ {error_message}")
        logs.append(failure_msg)

        return ProcessingResponse(status="error", message=error_message, logs=logs)


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the service is running."""
    return {
        "status": "healthy",
        "message": "Service is running",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
