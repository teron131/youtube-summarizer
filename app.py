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

from youtube_summarizer.summarizer import is_youtube_url, clean_youtube_url, summarize_video
from youtube_summarizer.utils import log_and_print
from youtube_summarizer.youtube_loader import youtube_loader

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
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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


class YouTubeProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL", min_length=1)
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class TextSummaryRequest(BaseModel):
    text: str = Field(..., description="Text content to summarize", min_length=1)


class URLValidationResponse(BaseModel):
    is_valid: bool = Field(description="Whether the URL is a valid YouTube URL")
    cleaned_url: Optional[str] = Field(description="Cleaned URL if valid")
    original_url: str = Field(description="Original URL provided")


class VideoInfoResponse(BaseModel):
    title: str
    author: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    url: str = Field(description="Cleaned YouTube URL")


class TranscriptResponse(BaseModel):
    title: str
    author: str
    transcript: str
    url: str = Field(description="Cleaned YouTube URL")
    processing_time: str


class SummaryResponse(BaseModel):
    title: str
    summary: str
    analysis: Optional[Dict[str, Any]] = None
    processing_time: str


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "YouTube Summarizer API",
        "version": "2.0.0",
        "description": "YouTube video processing with transcription & summarization",
        "docs": "/api/docs",
        "health": "/api/health",
        "endpoints": {
            "validate_url": "/api/validate-url",
            "video_info": "/api/video-info", 
            "transcript": "/api/transcript",
            "summary": "/api/summary",
            "process": "/api/process"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "YouTube Summarizer API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.post("/api/validate-url", response_model=URLValidationResponse)
async def validate_youtube_url(request: YouTubeRequest):
    """Validate and clean YouTube URL."""
    try:
        is_valid = is_youtube_url(request.url)
        cleaned_url = clean_youtube_url(request.url) if is_valid else None
        
        return URLValidationResponse(
            is_valid=is_valid,
            cleaned_url=cleaned_url,
            original_url=request.url
        )
    except Exception as e:
        log_and_print(f"❌ URL validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL validation failed: {str(e)}")


@app.post("/api/video-info", response_model=VideoInfoResponse)
async def get_video_info(request: YouTubeRequest):
    """Extract basic video information without processing."""
    try:
        # Validate and clean URL
        if not is_youtube_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
        
        cleaned_url = clean_youtube_url(request.url)
        log_and_print(f"📋 Extracting video info for: {cleaned_url}")
        
        # Use pytubefix to get basic metadata
        from pytubefix import YouTube
        youtube = YouTube(cleaned_url, client="WEB")
        
        metadata = {
            "title": youtube.title,
            "author": youtube.author,
            "duration": f"{getattr(youtube, 'length', 0)}s" if hasattr(youtube, 'length') else None,
            "thumbnail": getattr(youtube, 'thumbnail_url', None),
            "view_count": getattr(youtube, 'views', None),
            "upload_date": str(getattr(youtube, 'publish_date', None)) if getattr(youtube, 'publish_date', None) else None,
            "url": cleaned_url
        }
        
        return VideoInfoResponse(**metadata)
    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Video info extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract video info: {str(e)}")


@app.post("/api/transcript", response_model=TranscriptResponse)
async def get_video_transcript(request: YouTubeRequest):
    """Extract video transcript without generating summary."""
    start_time = datetime.now()
    
    try:
        # Validate and clean URL
        if not is_youtube_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
        
        cleaned_url = clean_youtube_url(request.url)
        log_and_print(f"📋 Extracting transcript for: {cleaned_url}")
        
        # Process video using the hybrid loader
        video_content = youtube_loader(cleaned_url)
        
        # Parse the content to extract metadata and transcript
        lines = video_content.split('\n')
        title = "Unknown Title"
        author = "Unknown Author"
        transcript = ""
        
        for i, line in enumerate(lines):
            if line.startswith("Title: "):
                title = line[7:]
            elif line.startswith("Author: "):
                author = line[8:]
            elif line.startswith("subtitle:"):
                transcript = '\n'.join(lines[i+1:])
                break
        
        processing_time = datetime.now() - start_time
        
        return TranscriptResponse(
            title=title,
            author=author,
            transcript=transcript,
            url=cleaned_url,
            processing_time=f"{processing_time.total_seconds():.1f}s"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Transcript extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract transcript: {str(e)}")


@app.post("/api/summary", response_model=SummaryResponse)
async def generate_text_summary(request: TextSummaryRequest):
    """Generate summary from provided text content."""
    start_time = datetime.now()
    
    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
        
        log_and_print("📋 Generating summary from provided text...")
        
        # Generate summary using the text
        analysis = summarize_video(request.text)
        
        # Convert structured analysis to summary text
        summary_parts = [
            f"**{analysis.title}**",
            "",
            "**Overall Summary:**",
            analysis.overall_summary,
            "",
            "**Key Takeaways:**"
        ]
        summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])
        
        if analysis.key_facts:
            summary_parts.extend(["", "**Key Facts:**"])
            summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])
        
        if analysis.chapters:
            summary_parts.extend(["", "**Chapters:**"])
            for chapter in analysis.chapters:
                summary_parts.extend([
                    f"**{chapter.header}**",
                    chapter.summary,
                    ""
                ])
        
        summary = '\n'.join(summary_parts)
        processing_time = datetime.now() - start_time
        
        return SummaryResponse(
            title=analysis.title,
            summary=summary,
            analysis={
                "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters],
                "key_facts": analysis.key_facts,
                "takeaways": analysis.takeaways,
                "overall_summary": analysis.overall_summary
            },
            processing_time=f"{processing_time.total_seconds():.1f}s"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.post("/api/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeProcessRequest):
    """
    Complete YouTube video processing with transcript extraction and optional summarization.
    This endpoint orchestrates the entire workflow from video info extraction to final summary generation.
    """
    start_time = datetime.now()
    logs = [f"🎬 Starting processing: {request.url}"]

    try:
        # Validate URL
        if not request.url.strip():
            raise HTTPException(status_code=400, detail="YouTube URL is required")

        if not is_youtube_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

        # Clean URL
        cleaned_url = clean_youtube_url(request.url)
        log_and_print(f"🔗 Cleaned URL: {cleaned_url}")
        logs.append(f"🔗 Cleaned URL: {cleaned_url}")

        # Process video using the hybrid loader
        log_and_print("📋 Step 1: Processing video with hybrid loader...")
        logs.append("📋 Step 1: Processing video with hybrid loader...")

        try:
            video_content = youtube_loader(cleaned_url)
            log_and_print("✅ Video content extracted successfully")
            logs.append("✅ Video content extracted successfully")
        except Exception as loader_error:
            error_msg = f"❌ Video processing failed: {str(loader_error)}"
            log_and_print(error_msg)
            logs.append(error_msg)
            raise HTTPException(status_code=400, detail=str(loader_error))

        # Parse the content to extract metadata and transcript
        lines = video_content.split('\n')
        title = "Unknown Title"
        author = "Unknown Author"
        transcript = ""
        
        for i, line in enumerate(lines):
            if line.startswith("Title: "):
                title = line[7:]
            elif line.startswith("Author: "):
                author = line[8:]
            elif line.startswith("subtitle:"):
                transcript = '\n'.join(lines[i+1:])
                break

        # Generate summary if requested and transcript is available
        summary = None
        analysis_data = None
        
        if request.generate_summary and transcript and not transcript.startswith("["):
            log_and_print("📋 Step 2: Generating summary...")
            logs.append("📋 Step 2: Generating summary...")

            if not os.getenv("GEMINI_API_KEY"):
                log_and_print("❌ GEMINI_API_KEY not configured")
                logs.append("❌ GEMINI_API_KEY not configured")
                summary = "[GEMINI_API_KEY not configured - please set your Gemini API key]"
            else:
                try:
                    full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{transcript}"
                    analysis = summarize_video(full_content)
                    
                    # Convert structured analysis to summary text
                    summary_parts = [
                        f"**{analysis.title}**",
                        "",
                        "**Overall Summary:**",
                        analysis.overall_summary,
                        "",
                        "**Key Takeaways:**"
                    ]
                    summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])
                    
                    if analysis.key_facts:
                        summary_parts.extend(["", "**Key Facts:**"])
                        summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])
                    
                    if analysis.chapters:
                        summary_parts.extend(["", "**Chapters:**"])
                        for chapter in analysis.chapters:
                            summary_parts.extend([
                                f"**{chapter.header}**",
                                chapter.summary,
                                ""
                            ])
                    
                    summary = '\n'.join(summary_parts)
                    
                    # Store structured analysis data
                    analysis_data = {
                        "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters],
                        "key_facts": analysis.key_facts,
                        "takeaways": analysis.takeaways,
                        "overall_summary": analysis.overall_summary
                    }
                    
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
            "title": title,
            "author": author,
            "transcript": transcript,
            "summary": summary,
            "analysis": analysis_data,
            "processing_time": f"{processing_time.total_seconds():.1f}s",
            "url": cleaned_url,
            "original_url": request.url,
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