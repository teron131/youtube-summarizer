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
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from youtube_summarizer.summarizer import (
    Analysis,
    GraphOutput,
    Quality,
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


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()

    log_and_print(f"📨 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    return response


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
    duration: str | None = None
    thumbnail: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    upload_date: str | None = None
    processing_time: str


class SummarizeRequest(BaseModel):
    """Summarization request."""

    content: str = Field(..., min_length=10, max_length=50000, description="Content to analyze")
    content_type: str = Field(default="url", pattern=r"^(url|transcript)$")

    # Model selection
    analysis_model: str = Field(default="google/gemini-2.5-pro", description="Model for analysis generation")
    quality_model: str = Field(default="google/gemini-2.5-flash", description="Model for quality evaluation")

    # Translation options
    enable_translation: bool = Field(default=False, description="Enable translation to target language")
    target_language: str = Field(default="en", description="Target language for translation", min_length=2, max_length=5)


class SummarizeResponse(BaseResponse):
    """Summarization response."""

    analysis: Analysis
    quality: Quality | None = None
    processing_time: str
    iteration_count: int = Field(default=1)

    # Model metadata
    analysis_model: str = Field(description="Model used for analysis")
    quality_model: str = Field(description="Model used for quality evaluation")

    # Translation metadata
    target_language: str | None = Field(default=None, description="Target language used for translation")
    enable_translation: bool = Field(default=False, description="Whether translation was enabled")


class ConfigurationResponse(BaseResponse):
    """Configuration response with available options."""

    available_models: dict[str, str] = Field(description="Available models for selection")
    supported_languages: dict[str, str] = Field(description="Supported languages for translation")
    default_analysis_model: str = Field(description="Default analysis model")
    default_quality_model: str = Field(description="Default quality model")
    default_target_language: str = Field(description="Default target language")


# ================================
# UTILITIES
# ================================


def validate_url(url: str) -> str:
    """Validate and clean YouTube URL with enhanced checks."""
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["empty_url"])

    url = url.strip()
    if len(url) > 2048:
        raise HTTPException(status_code=400, detail="URL too long (max 2048 characters)")

    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["invalid_url"])

    return clean_youtube_url(url)


def validate_content(content: str) -> str:
    """Validate content for summarization requests."""
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    content = content.strip()
    if len(content) < 10:
        raise HTTPException(status_code=400, detail="Content too short (minimum 10 characters)")

    if len(content) > 50000:
        raise HTTPException(status_code=400, detail="Content too long (maximum 50,000 characters)")

    return content


def parse_scraper_result(result) -> dict[str, Any]:
    """Parse scraper result to dict with error handling."""
    try:
        # Handle both Pydantic objects and raw dict responses
        if hasattr(result, "model_dump"):
            # Pydantic object - use model_dump() to get all fields including properties
            result_dict = result.model_dump()
        else:
            # Raw dict
            result_dict = result if isinstance(result, dict) else {}

        # Map strictly to canonical fields defined in youtube_scrapper.py docs
        like_count = result_dict.get("likeCountInt")
        upload_date = result_dict.get("publishDateText")

        # Extract chapters if available
        chapters = []
        if result_dict.get("chapters"):
            chapters = [{"title": chapter.get("title", ""), "timeDescription": chapter.get("timeDescription", ""), "startSeconds": chapter.get("startSeconds", 0)} for chapter in result_dict["chapters"]]

        # Get transcript with proper fallback chain
        transcript = ""
        if hasattr(result, "parsed_transcript"):
            # If it's a Pydantic object, use the parsed_transcript property
            transcript = result.parsed_transcript
        elif result_dict.get("parsed_transcript"):
            # If it's in the dict (from model_dump), use it
            transcript = result_dict["parsed_transcript"]
        elif result_dict.get("transcript_only_text"):
            # Fallback to transcript_only_text
            transcript = result_dict["transcript_only_text"]
        elif hasattr(result, "transcript_only_text"):
            # Fallback for object attribute
            transcript = getattr(result, "transcript_only_text", "")

        # Ensure transcript is not empty - provide fallback if needed
        if not transcript or not transcript.strip():
            transcript = "Transcript not available for this video."

        return {
            "url": result_dict.get("url", ""),
            "title": result_dict.get("title", "Unknown Title") or "Unknown Title",
            "author": result_dict.get("channel", {}).get("title", "Unknown Author") if isinstance(result_dict.get("channel"), dict) else "Unknown Author",
            "transcript": transcript,
            "duration": result_dict.get("durationFormatted"),
            "thumbnail": result_dict.get("thumbnail"),
            "view_count": result_dict.get("viewCountInt"),
            "like_count": like_count,
            "upload_date": upload_date,
            "chapters": chapters,  # Include chapters in parsed result
        }
    except Exception as e:
        # Fallback parsing for malformed results
        log_and_print(f"⚠️ Warning: Error parsing scraper result: {str(e)}")

        # Extract chapters from fallback result if available
        chapters = []
        if hasattr(result, "chapters") and getattr(result, "chapters", []):
            chapters = [{"title": getattr(chapter, "title", ""), "timeDescription": getattr(chapter, "timeDescription", ""), "startSeconds": getattr(chapter, "startSeconds", 0)} for chapter in getattr(result, "chapters", [])]

        # Get transcript with fallback for error case
        transcript = ""
        if hasattr(result, "parsed_transcript"):
            transcript = result.parsed_transcript
        elif hasattr(result, "transcript_only_text"):
            transcript = getattr(result, "transcript_only_text", "")
        else:
            transcript = "Transcript not available for this video."

        return {
            "url": getattr(result, "url", ""),
            "title": getattr(result, "title", "Unknown Title"),
            "author": "Unknown Author",
            "transcript": transcript,
            "duration": getattr(result, "durationFormatted", None),
            "thumbnail": getattr(result, "thumbnail", None),
            "view_count": getattr(result, "viewCountInt", None),
            "like_count": getattr(result, "likeCountInt", None),
            # Best effort; do not transform
            "upload_date": getattr(result, "publishDateText", None),
            "chapters": chapters,  # Include chapters in fallback
        }


async def run_async_task(func, *args, timeout: float = TIMEOUT_LONG):
    """Execute blocking function asynchronously with timeout."""
    try:
        return await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, func, *args), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Request timed out after {timeout} seconds")
    except Exception as e:
        # Re-raise HTTPException as-is, wrap others
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:100]}")


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
        "health": "/health",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check with environment status",
            "GET /config": "Get available models and languages",
            "POST /scrap": "Extract video metadata and transcript",
            "POST /summarize": "Full LangGraph workflow analysis",
            "POST /stream-summarize": "Streaming analysis with progress",
        },
        "timestamp": datetime.now().isoformat(),
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


@app.get("/config", response_model=ConfigurationResponse)
async def get_configuration():
    """Get available models and languages for frontend configuration."""
    from youtube_summarizer.summarizer import (
        ANALYSIS_MODEL,
        QUALITY_MODEL,
        TARGET_LANGUAGE,
    )

    # Simple configuration for API response
    available_models = {
        "google/gemini-2.5-pro": "Gemini 2.5 Pro (Recommended)",
        "google/gemini-2.5-flash": "Gemini 2.5 Flash (Fast)",
        "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    }

    supported_languages = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "ru": "Russian",
    }

    return ConfigurationResponse(
        status="success",
        message="Configuration retrieved successfully",
        available_models=available_models,
        supported_languages=supported_languages,
        default_analysis_model=ANALYSIS_MODEL,
        default_quality_model=QUALITY_MODEL,
        default_target_language=TARGET_LANGUAGE,
    )


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

        return ScrapResponse(status="success", message="Video scraped successfully", url=data["url"], title=data["title"], author=data["author"], transcript=data["transcript"], duration=data["duration"], thumbnail=data["thumbnail"], view_count=data["view_count"], like_count=data["like_count"], upload_date=data["upload_date"], processing_time=get_processing_time(start_time))
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
        from youtube_summarizer.summarizer import (
            GraphInput,
            GraphOutput,
            create_compiled_graph,
        )

        # Validate content before processing
        validated_content = validate_content(request.content)

        # Extract chapters if content is a YouTube URL
        chapters = []
        if is_youtube_url(validated_content):
            try:
                print(f"🔗 Detected YouTube URL, scraping to get chapters...")
                from youtube_summarizer.youtube_scrapper import scrap_youtube

                scrap_result = await run_async_task(scrap_youtube, validated_content)
                parsed_data = parse_scraper_result(scrap_result)
                chapters = parsed_data.get("chapters", [])
                print(f"📋 Found {len(chapters)} chapters in YouTube video")
                if chapters:
                    print(f"📝 Chapter titles: {[ch['title'] for ch in chapters[:3]]}{'...' if len(chapters) > 3 else ''}")
            except Exception as e:
                print(f"⚠️ Failed to extract chapters from YouTube URL: {str(e)}")
                chapters = []

        # Create GraphInput with new parameters
        graph_input = GraphInput(
            transcript_or_url=validated_content,
            chapters=chapters,
            enable_translation=request.enable_translation,
            target_language=request.target_language,
            analysis_model=request.analysis_model,
            quality_model=request.quality_model,
        )

        # Use the LangGraph workflow
        graph = create_compiled_graph()
        result = await run_async_task(graph.invoke, graph_input)

        # Convert to GraphOutput for type safety
        graph_output: GraphOutput = GraphOutput.model_validate(result)

        return SummarizeResponse(
            status="success",
            message="Analysis completed successfully",
            analysis=graph_output.analysis,
            quality=graph_output.quality,
            processing_time=get_processing_time(start_time),
            iteration_count=graph_output.iteration_count,
            target_language=request.target_language if request.enable_translation else None,
            enable_translation=request.enable_translation,
            analysis_model=request.analysis_model,
            quality_model=request.quality_model,
        )
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
            # Validate content before processing
            validated_content = validate_content(request.content)

            # Extract chapters if content is a YouTube URL
            chapters = []
            if is_youtube_url(validated_content):
                try:
                    print(f"🔗 Detected YouTube URL in streaming, scraping to get chapters...")
                    from youtube_summarizer.youtube_scrapper import scrap_youtube

                    scrap_result = await run_async_task(scrap_youtube, validated_content)
                    parsed_data = parse_scraper_result(scrap_result)
                    chapters = parsed_data.get("chapters", [])
                    print(f"📋 Found {len(chapters)} chapters in YouTube video for streaming")
                except Exception as e:
                    print(f"⚠️ Failed to extract chapters from YouTube URL in streaming: {str(e)}")
                    chapters = []

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for client processing

            # Create GraphInput with new parameters for streaming
            from youtube_summarizer.summarizer import GraphInput, create_compiled_graph

            graph_input = GraphInput(
                transcript_or_url=validated_content,
                chapters=chapters,
                enable_translation=request.enable_translation,
                target_language=request.target_language,
                analysis_model=request.analysis_model,
                quality_model=request.quality_model,
            )

            graph = create_compiled_graph()

            chunk_count = 0
            for chunk in graph.stream(graph_input, stream_mode="values"):
                try:
                    # Handle both dictionary (from LangGraph) and Pydantic model cases
                    if isinstance(chunk, dict):
                        chunk_dict = chunk.copy()
                    else:
                        chunk_dict = chunk.model_dump()

                    # Ensure all nested Pydantic models are serialized to dict
                    def serialize_nested(obj):
                        if isinstance(obj, dict):
                            return {k: serialize_nested(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [serialize_nested(item) for item in obj]
                        elif hasattr(obj, "model_dump"):
                            return obj.model_dump()
                        else:
                            return obj

                    chunk_dict = serialize_nested(chunk_dict)

                    # Filter out large data fields that can break JSON streaming
                    # Remove transcript data from streaming chunks to prevent JSON parsing errors
                    filtered_chunk = {}
                    for key, value in chunk_dict.items():
                        # Skip large text fields that can cause JSON parsing issues
                        if key in ["transcript_or_url"] and isinstance(value, str) and len(value) > 1000:
                            # Truncate large transcript data to prevent JSON issues
                            filtered_chunk[key] = value[:500] + "...[truncated for streaming]"
                        elif key in ["analysis", "quality"] and isinstance(value, dict):
                            # Keep analysis and quality data but limit nested text fields
                            filtered_analysis = {}
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str) and len(sub_value) > 2000:
                                    filtered_analysis[sub_key] = sub_value[:1000] + "...[truncated]"
                                else:
                                    filtered_analysis[sub_key] = sub_value
                            filtered_chunk[key] = filtered_analysis
                        else:
                            filtered_chunk[key] = value

                    filtered_chunk["timestamp"] = datetime.now().isoformat()
                    filtered_chunk["chunk_number"] = chunk_count

                    # Validate JSON before yielding
                    json_str = json.dumps(filtered_chunk, ensure_ascii=False)
                    if len(json_str) > 50000:  # Skip extremely large chunks
                        print(f"⚠️ Skipping oversized chunk ({len(json_str)} chars)")
                        continue

                    yield f"data: {json_str}\n\n"
                    chunk_count += 1

                    # Adaptive delay based on chunk count to prevent overwhelming client
                    delay = min(0.1, 0.05 + (chunk_count * 0.01))
                    await asyncio.sleep(delay)

                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    print(f"⚠️ Failed to serialize chunk {chunk_count}: {str(e)}")
                    # Skip problematic chunks but continue processing
                    chunk_count += 1
                    continue

            # Send completion message
            completion_data = {"type": "complete", "message": "Analysis completed", "processing_time": get_processing_time(start_time), "total_chunks": chunk_count, "timestamp": datetime.now().isoformat()}
            yield f"data: {json.dumps(completion_data)}\n\n"

        except Exception as e:
            log_and_print(f"❌ Streaming failed: {str(e)}")
            error_data = {"type": "error", "message": str(e)[:100], "timestamp": datetime.now().isoformat()}
            try:
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception as json_error:
                # Fallback if even error serialization fails
                log_and_print(f"❌ Failed to serialize error data: {str(json_error)}")
                yield f'data: {{"type": "error", "message": "Streaming failed", "timestamp": "{datetime.now().isoformat()}"}}\n\n'

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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
