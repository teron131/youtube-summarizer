"""
YouTube Summarizer FastAPI Application
-------------------------------------

This FastAPI application provides an API to download, transcribe, and summarize
YouTube videos. Backend-only package for programmatic use.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    extract_video_metadata,
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
    title="YouTube Summarizer API",
    description="YouTube video processing with transcription & summarization",
    version="1.0.0",
)

# CORS configuration for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for your use case
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class YouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL", min_length=1)
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class VideoInfoResponse(BaseModel):
    title: str
    author: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    upload_date: Optional[str] = None


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


class ProcessingStatus(BaseModel):
    stage: str
    progress: int
    message: str


# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "YouTube Summarizer API is running",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/video-info", response_model=VideoInfoResponse)
async def get_video_info(request: YouTubeRequest):
    """Extract basic video information without processing."""
    try:
        log_and_print(f"📋 Extracting video info for: {request.url}")
        info = extract_video_info(request.url)
        metadata = extract_video_metadata(info)
        return VideoInfoResponse(**metadata)
    except Exception as e:
        log_and_print(f"❌ Video info extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract video info: {str(e)}")


@app.post("/api/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeRequest):
    """
    Process YouTube video with robust error handling and visible logging.
    This endpoint orchestrates the entire workflow from video info extraction
    to final summary generation.
    """
    start_time = datetime.now()
    logs = [f"🎬 Starting processing: {request.url}"]

    try:
        # Validate URL
        if not request.url.strip():
            raise HTTPException(status_code=400, detail="YouTube URL is required")

        log_and_print("📋 Step 1: Extracting video info...")
        logs.append("📋 Step 1: Extracting video info...")
        info = extract_video_info(request.url)

        video_metadata = VideoInfoResponse(**extract_video_metadata(info))
        log_and_print(f"✅ Video found: {video_metadata.title} by {video_metadata.author}")
        logs.append(f"✅ Video found: {video_metadata.title} by {video_metadata.author}")

        log_and_print("📋 Step 2: Checking for existing captions...")
        logs.append("📋 Step 2: Checking for existing captions...")
        subtitle = get_subtitle_from_captions(info)

        if subtitle:
            log_and_print("✅ Found existing captions - skipping transcription")
            logs.append("✅ Found existing captions - skipping transcription")
            formatted_subtitle = simple_format_subtitle(subtitle)
        else:
            log_and_print("🎯 No captions found - proceeding with transcription")
            logs.append("🎯 No captions found - proceeding with transcription")

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
                    formatted_subtitle = "[FAL_KEY not configured - please set your FAL API key]"
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

        # Generate summary if requested and transcript is available
        summary = None
        if request.generate_summary and not formatted_subtitle.startswith("["):
            log_and_print("📋 Step 5: Generating summary...")
            logs.append("📋 Step 5: Generating summary...")

            if not os.getenv("GEMINI_API_KEY"):
                log_and_print("❌ GEMINI_API_KEY not configured")
                logs.append("❌ GEMINI_API_KEY not configured")
                summary = "[GEMINI_API_KEY not configured - please set your Gemini API key]"
            else:
                try:
                    full_content = f"Title: {video_metadata.title}\nAuthor: {video_metadata.author}\nTranscript:\n{formatted_subtitle}"
                    summary = quick_summary(full_content)
                    log_and_print("✅ Summary generated")
                    logs.append("✅ Summary generated")
                except Exception as summary_error:
                    error_msg = f"❌ Summary generation failed: {str(summary_error)}"
                    log_and_print(error_msg)
                    logs.append(error_msg)
                    summary = f"[Summary generation failed: {str(summary_error)}]"

        processing_time = datetime.now() - start_time
        completion_msg = f"✅ Processing completed in {processing_time.total_seconds():.1f}s"
        log_and_print(completion_msg)
        logs.append(completion_msg)

        result_data = {
            "title": video_metadata.title,
            "author": video_metadata.author,
            "duration": video_metadata.duration,
            "thumbnail": video_metadata.thumbnail,
            "transcript": formatted_subtitle,
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

    except HTTPException:
        raise
    except Exception as e:
        processing_time = datetime.now() - start_time
        error_message = f"Processing error: {str(e)}"
        failure_msg = f"💔 Failed after {processing_time.total_seconds():.1f}s"
        log_and_print(f"❌ {error_message}")
        log_and_print(failure_msg)
        logs.append(f"❌ {error_message}")
        logs.append(failure_msg)

        return ProcessingResponse(status="error", message=error_message, logs=logs)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    log_and_print(f"🚀 Starting YouTube Summarizer API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
