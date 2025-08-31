"""
YouTube Summarizer FastAPI Application
=====================================

Optimized FastAPI application for YouTube video processing and summarization.
Uses the youtube_summarizer package functions directly for clean, efficient backend deployment.

## 🔧 Configuration
Set environment variables:
- GEMINI_API_KEY - For AI summarization (required)
- APIFY_API_KEY - For YouTube scraping (optional fallback)
- PORT - Server port (default: 8080)
- HOST - Server host (default: 0.0.0.0)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from youtube_summarizer.summarizer import (
    Analysis,
    Quality,
    WorkflowOutput,
    stream_summarize_video,
)
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url, log_and_print
from youtube_summarizer.youtube_scrapper import scrap_youtube

load_dotenv()

# ================================
# CONFIGURATION
# ================================

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"
API_DESCRIPTION = "Optimized YouTube video processing and summarization"

# Simplified error messages
ERROR_MESSAGES = {
    "invalid_url": "Invalid YouTube URL",
    "empty_url": "URL is required",
    "processing_failed": "Processing failed",
    "api_quota_exceeded": "API quota exceeded",
    "config_missing": "Required API key missing",
}

# Timeout settings
TIMEOUT_LONG = 300.0  # 5 minutes for AI processing


# ================================
# FASTAPI APPLICATION SETUP
# ================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
)

# Simple CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ================================
# PYDANTIC MODELS
# ================================


class BaseResponse(BaseModel):
    """Base response model."""

    status: str = Field(description="Response status: success or error")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class YouTubeRequest(BaseModel):
    """YouTube URL request."""

    url: str = Field(..., min_length=10, max_length=2048, description="YouTube video URL")


class ScrapResponse(BaseResponse):
    """Video scraping response."""

    url: str
    title: str
    author: str
    transcript: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    processing_time: str


class SummarizeRequest(BaseModel):
    """Summarization request."""

    content: str = Field(..., min_length=10, max_length=50000, description="Content to analyze")
    content_type: str = Field(default="url", pattern=r"^(url|transcript)$")


class SummarizeResponse(BaseResponse):
    """Summarization response."""

    analysis: Analysis
    quality: Optional[Quality] = None
    processing_time: str
    iteration_count: int = Field(default=1)


# ================================
# UTILITIES
# ================================


def validate_url(url: str) -> str:
    """Validate and clean YouTube URL."""
    if not url.strip():
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["empty_url"])
    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["invalid_url"])
    return clean_youtube_url(url)


def parse_scraper_result(result) -> Dict[str, Any]:
    """Parse scraper result to dict with error handling."""
    try:
        # Handle both Pydantic objects and raw dict responses
        if hasattr(result, "model_dump"):
            # Pydantic object
            result_dict = result.model_dump()
        else:
            # Raw dict
            result_dict = result if isinstance(result, dict) else {}

        return {
            "url": result_dict.get("url", ""),
            "title": result_dict.get("title", "Unknown Title") or "Unknown Title",
            "author": result_dict.get("channel", {}).get("title", "Unknown Author") if isinstance(result_dict.get("channel"), dict) else "Unknown Author",
            "transcript": result_dict.get("parsed_transcript", result_dict.get("transcript_only_text", "")),
            "duration": result_dict.get("durationFormatted"),
            "thumbnail": result_dict.get("thumbnail"),
            "view_count": result_dict.get("viewCountInt"),
        }
    except Exception as e:
        # Fallback parsing for malformed results
        log_and_print(f"⚠️ Warning: Error parsing scraper result: {str(e)}")
        return {
            "url": getattr(result, "url", ""),
            "title": getattr(result, "title", "Unknown Title"),
            "author": "Unknown Author",
            "transcript": getattr(result, "transcript_only_text", ""),
            "duration": None,
            "thumbnail": getattr(result, "thumbnail", None),
            "view_count": getattr(result, "viewCountInt", None),
        }


async def run_async_task(func, *args):
    """Execute blocking function asynchronously."""
    return await asyncio.get_event_loop().run_in_executor(None, func, *args)


def get_processing_time(start_time: datetime) -> str:
    """Calculate and format processing time."""
    return f"{(datetime.now() - start_time).total_seconds():.1f}s"


def create_error_response(status_code: int, error_type: str, exception: Exception = None) -> HTTPException:
    """Create standardized error response."""
    message = ERROR_MESSAGES.get(error_type, "Unknown error")
    if exception:
        message += f": {str(exception)[:100]}"
    raise HTTPException(status_code=status_code, detail=message)


# ================================
# API ENDPOINTS
# ================================


@app.get("/")
async def root():
    """API information and health check."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "endpoints": {
            "scrap": "/scrap - Extract video metadata and transcript",
            "summarize": "/summarize - Full LangGraph workflow analysis",
            "stream-summarize": "/stream-summarize - Streaming analysis with progress",
        },
    }


# ================================
# CORE ENDPOINTS
# ================================


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "message": f"{API_TITLE} is running",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
        "environment": {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "apify_configured": bool(os.getenv("APIFY_API_KEY"))},
    }


@app.post("/scrap", response_model=ScrapResponse)
async def scrap_video(request: YouTubeRequest):
    """Extract video metadata and transcript."""
    start_time = datetime.now()

    if not os.getenv("APIFY_API_KEY"):
        create_error_response(500, "config_missing")

    try:
        url = validate_url(request.url)
        result = await run_async_task(scrap_youtube, url)
        data = parse_scraper_result(result)

        return ScrapResponse(status="success", message="Video scraped successfully", url=data["url"], title=data["title"], author=data["author"], transcript=data["transcript"], duration=data["duration"], thumbnail=data["thumbnail"], view_count=data["view_count"], processing_time=get_processing_time(start_time))
    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Scraping failed: {str(e)}")
        if "quota" in str(e).lower():
            create_error_response(429, "api_quota_exceeded", e)
        elif "400" in str(e) or "Invalid" in str(e):
            create_error_response(400, "invalid_url", e)
        else:
            create_error_response(500, "processing_failed", e)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate AI analysis using LangGraph workflow."""
    if not os.getenv("GEMINI_API_KEY"):
        create_error_response(500, "config_missing")

    start_time = datetime.now()

    try:
        from youtube_summarizer.summarizer import WorkflowInput, create_compiled_graph

        graph = create_compiled_graph()
        result = await run_async_task(graph.invoke, WorkflowInput(transcript_or_url=request.content))
        workflow_output = WorkflowOutput.model_validate(result)

        return SummarizeResponse(status="success", message="Analysis completed successfully", analysis=workflow_output.analysis, quality=workflow_output.quality, processing_time=get_processing_time(start_time), iteration_count=workflow_output.iteration_count)
    except Exception as e:
        log_and_print(f"❌ Analysis failed: {str(e)}")
        create_error_response(500, "processing_failed", e)


@app.post("/stream-summarize")
async def stream_summarize(request: SummarizeRequest):
    """Stream analysis with real-time progress updates."""
    if not os.getenv("GEMINI_API_KEY"):
        create_error_response(500, "config_missing")

    async def generate_stream():
        start_time = datetime.now()

        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"

            for chunk in stream_summarize_video(request.content):
                chunk_dict = chunk.model_dump()
                chunk_dict["timestamp"] = datetime.now().isoformat()
                yield f"data: {json.dumps(chunk_dict)}\n\n"
                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'type': 'complete', 'message': 'Analysis completed', 'processing_time': get_processing_time(start_time), 'timestamp': datetime.now().isoformat()})}\n\n"

        except Exception as e:
            log_and_print(f"❌ Streaming failed: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:100], 'timestamp': datetime.now().isoformat()})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    log_and_print(f"🚀 Starting {API_TITLE} v{API_VERSION} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
