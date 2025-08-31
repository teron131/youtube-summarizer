"""
YouTube Summarizer FastAPI Application
=====================================

This FastAPI application provides a comprehensive API for YouTube video processing,
including scraping, transcription, and AI-powered summarization.

## 🔄 Processing Workflow

The application uses a multi-tier fallback approach for optimal results:

### Tier 1: Apify YouTube Scraper API
- Uses professional YouTube scraping API to extract video metadata and transcripts
- Fastest and most reliable method when API quota is available
- Supports automatic transcript extraction and chapter detection

### Tier 2: Gemini Direct Processing
- Directly processes YouTube URLs using Google's Gemini AI
- Fallback when API fails or quota is exhausted
- Provides analysis even for problematic videos

## 📊 API Endpoints

- `/validate-url` - Validate YouTube URLs
- `/video-info` - Extract video metadata
- `/transcript` - Get video transcripts
- `/summary` - Generate text summaries
- `/process` - Complete processing pipeline
- `/generate` - Master endpoint with all capabilities

## 🔧 Configuration

Set environment variables:
- `APIFY_API_KEY` - For YouTube scraping API
- `GEMINI_API_KEY` - For AI summarization
- `FAL_KEY` - For audio transcription (legacy fallback)
- `PORT` - Server port (default: 8080)
- `HOST` - Server host (default: 0.0.0.0)

Backend-only package for programmatic use and frontend integration.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from httpx import RemoteProtocolError
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from youtube_summarizer.summarizer import (
    Analysis,
    Quality,
    WorkflowOutput,
    stream_summarize_video,
    summarize_video,
)
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url, log_and_print
from youtube_summarizer.youtube_scrapper import YouTubeScrapperResult, scrap_youtube

load_dotenv()

# ================================
# CONSTANTS & CONFIGURATION
# ================================

API_VERSION = "2.1.0"  # Incremented for optimizations
API_TITLE = "YouTube Summarizer API"
API_DESCRIPTION = "YouTube video processing with transcription & summarization"

# Enhanced configuration
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB limit
CACHE_TTL = 3600  # 1 hour cache
RATE_LIMIT_REQUESTS = 100  # per minute
TIMEOUT_SHORT = 30.0  # seconds for quick operations
TIMEOUT_LONG = 300.0  # seconds for AI processing (reduced from 600)

# Error messages
ERROR_MESSAGES = {
    "invalid_url": "Invalid YouTube URL format",
    "empty_url": "YouTube URL is required",
    "malicious_url": "URL appears to be malicious or invalid",
    "gemini_not_configured": "GEMINI_API_KEY not configured",
    "apify_not_configured": "APIFY_API_KEY not configured",
    "fal_not_configured": "FAL_KEY not configured (legacy fallback)",
    "video_too_long": "Video is too long for processing. Please try with a shorter video or use time segments.",
    "processing_failed": "All processing methods failed",
    "api_quota_exceeded": "YouTube scraping API quota exceeded. Please try again later.",
    "rate_limit_exceeded": "Too many requests. Please try again later.",
    "request_too_large": "Request size exceeds maximum allowed limit",
    "timeout_error": "Request timed out. The video may be too long for processing.",
}

# Processing status indicators
PROCESSING_STATUS = {
    "pending": "pending",
    "success": "success",
    "failed": "failed",
}

# ================================
# ENHANCED LOGGING
# ================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with request tracking."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request details
        logger.info(f"🔄 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")

        response = await call_next(request)

        # Log response details
        process_time = time.time() - start_time
        logger.info(f"✅ {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")

        return response


# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("youtube_summarizer.log", mode="a") if os.getenv("LOG_TO_FILE") else logging.NullHandler()],
)
logger = logging.getLogger(__name__)

# ================================
# CACHING UTILITIES
# ================================


class CacheManager:
    """Simple in-memory cache with TTL support."""

    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> None:
        """Set value in cache with TTL."""
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


# Global cache instance
cache = CacheManager()


def cache_response(ttl: int = CACHE_TTL):
    """Decorator for caching function responses."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"📋 Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.info(f"💾 Cached result for {func.__name__}")
            return result

        return wrapper

    return decorator


# ================================
# RATE LIMITING
# ================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = RATE_LIMIT_REQUESTS):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.clients: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries
        if client_ip in self.clients:
            self.clients[client_ip] = [req_time for req_time in self.clients[client_ip] if current_time - req_time < 60]  # Keep only last minute
        else:
            self.clients[client_ip] = []

        # Check rate limit
        if len(self.clients[client_ip]) >= self.requests_per_minute:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail=ERROR_MESSAGES["rate_limit_exceeded"])

        # Add current request
        self.clients[client_ip].append(current_time)

        return await call_next(request)


# ================================
# FASTAPI APPLICATION SETUP
# ================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# Rate limiting
app.add_middleware(RateLimitMiddleware, requests_per_minute=RATE_LIMIT_REQUESTS)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# CORS with more restrictive defaults (can be overridden via env vars)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://youtube-summarizer-ui-teron131.up.railway.app")
allowed_origins = os.getenv("ALLOWED_ORIGINS", f"http://localhost:3000,http://localhost:8080,{FRONTEND_URL}").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Enhanced timeout middleware with progress feedback
@app.middleware("http")
async def enhanced_timeout_middleware(request: Request, call_next):
    """Enhanced timeout handling with different timeouts for different endpoints."""

    # Determine timeout based on endpoint
    timeout = TIMEOUT_SHORT

    # Streaming endpoints get unlimited timeout (they manage their own connection)
    if any(
        path in str(request.url)
        for path in [
            "/stream-summarize",
            "/stream-process",
            "/api/stream-summarize",
            "/api/stream-process",
        ]
    ):
        # No timeout for streaming endpoints
        return await call_next(request)

    if any(
        path in str(request.url)
        for path in [
            "/process",
            "/generate",
            "/summary",
            "/summarize",
            "/api/process",
            "/api/generate",
            "/api/summary",
            "/api/summarize",
        ]
    ):
        timeout = TIMEOUT_LONG

    try:
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        logger.error(f"⏰ Request timeout after {timeout}s for {request.url}")
        return Response(content='{"status": "error", "message": "' + ERROR_MESSAGES["timeout_error"] + '"}', status_code=408, media_type="application/json")


# ================================
# ENHANCED PYDANTIC MODELS
# ================================


class YouTubeRequest(BaseModel):
    """Basic YouTube URL request model with validation."""

    url: str = Field(..., description="YouTube video URL", min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v):
        """Enhanced URL validation."""
        if not v.strip():
            raise ValueError(ERROR_MESSAGES["empty_url"])

        # Basic malicious URL patterns
        malicious_patterns = ["javascript:", "data:", "file:", "ftp:"]
        if any(pattern in v.lower() for pattern in malicious_patterns):
            raise ValueError(ERROR_MESSAGES["malicious_url"])

        return v.strip()


class YouTubeProcessRequest(BaseModel):
    """Extended request model for processing with options."""

    url: str = Field(..., description="YouTube video URL", min_length=1, max_length=2048)
    generate_summary: bool = Field(default=True, description="Generate AI summary")

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v):
        return YouTubeRequest.validate_youtube_url(v)


class TextSummaryRequest(BaseModel):
    """Request model for text-only summarization."""

    text: str = Field(..., description="Text content to summarize", min_length=10, max_length=100000)


class GenerateRequest(BaseModel):
    """Comprehensive request model for the master generate endpoint."""

    url: str = Field(..., description="YouTube video URL or 'example' for demo", max_length=2048)
    include_transcript: bool = Field(default=True, description="Include full transcript")
    include_summary: bool = Field(default=True, description="Include AI summary")
    include_analysis: bool = Field(default=True, description="Include structured analysis")
    include_metadata: bool = Field(default=True, description="Include video metadata")


# Enhanced response models with better error handling
class BaseResponse(BaseModel):
    """Base response model with consistent error handling."""

    status: str = Field(description="Response status: success, error, or partial")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class URLValidationResponse(BaseResponse):
    """Response model for URL validation."""

    is_valid: bool = Field(description="Whether the URL is a valid YouTube URL")
    cleaned_url: Optional[str] = Field(description="Cleaned URL if valid")
    original_url: str = Field(description="Original URL provided")


class VideoInfoResponse(BaseModel):
    """Response model for video metadata."""

    url: str = Field(description="Cleaned YouTube URL")
    title: str
    author: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    upload_date: Optional[str] = None


class TranscriptResponse(BaseResponse):
    """Response model for transcript extraction."""

    url: str = Field(description="Cleaned YouTube URL")
    title: str
    author: str
    transcript: str
    processing_time: str
    source: str = Field(description="Data source: apify_api, gemini_direct")


class SummaryResponse(BaseResponse):
    """Response model for text summarization."""

    title: str
    summary: str
    analysis: Optional[Dict[str, Any]] = None
    processing_time: str


class ProcessingResponse(BaseResponse):
    """Response model for complete video processing."""

    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


class GenerateResponse(BaseResponse):
    """Comprehensive response model for the master generate endpoint."""

    video_info: Optional[VideoInfoResponse] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    processing_details: Dict[str, str] = {}
    logs: List[str] = []


class ScrapRequest(BaseModel):
    """Request model for the scrap endpoint."""

    url: str = Field(..., description="YouTube video URL", min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v):
        return YouTubeRequest.validate_youtube_url(v)


class ScrapResponse(BaseResponse):
    """Response model for the scrap endpoint."""

    cleaned_url: str = Field(description="Cleaned YouTube URL")
    video_info: VideoInfoResponse = Field(description="Video metadata")
    transcript: str = Field(description="Video transcript")
    processing_time: str = Field(description="Processing duration")


class SummarizeRequest(BaseModel):
    """Request model for the summarize endpoint."""

    content: str = Field(..., description="Content to analyze (URL or transcript)", min_length=10, max_length=100000)
    content_type: str = Field(default="url", description="Type of content: 'url' or 'transcript'")


class SummarizeResponse(BaseResponse):
    """Response model for the summarize endpoint using full LangGraph workflow."""

    analysis: Analysis = Field(description="Structured analysis data")
    quality: Optional[Quality] = Field(default=None, description="Quality assessment")
    processing_time: str = Field(description="Processing duration")
    iteration_count: int = Field(default=1, description="Number of refinement iterations")


class StreamSummarizeRequest(BaseModel):
    """Request model for the streaming summarize endpoint."""

    content: str = Field(..., description="Content to analyze (URL or transcript)", min_length=10, max_length=100000)
    content_type: str = Field(default="url", description="Type of content: 'url' or 'transcript'")


# ================================
# ENHANCED HELPER FUNCTIONS
# ================================


@lru_cache(maxsize=128)
def validate_url(url: str) -> str:
    """
    Validate and clean YouTube URL with caching.

    Args:
        url: Raw YouTube URL

    Returns:
        Cleaned YouTube URL

    Raises:
        HTTPException: If URL is invalid
    """
    if not url.strip():
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["empty_url"])

    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["invalid_url"])

    return clean_youtube_url(url)


def create_error_response(status_code: int, message: str, details: Optional[Dict] = None) -> HTTPException:
    """Standardized error response creator."""
    error_detail = {"error": message, "timestamp": datetime.now().isoformat()}
    if details:
        error_detail.update(details)
    return HTTPException(status_code=status_code, detail=error_detail)


def parse_video_content(scrapper_result: YouTubeScrapperResult) -> Tuple[str, str, str]:
    """
    Parse video content from YouTubeScrapperResult.

    Args:
        scrapper_result: YouTubeScrapperResult from scrap_youtube

    Returns:
        Tuple of (title, author, transcript)
    """
    title = scrapper_result.title or "Unknown Title"
    author = scrapper_result.channel.title if scrapper_result.channel and scrapper_result.channel.title else "Unknown Author"
    transcript = scrapper_result.parsed_transcript

    return title, author, transcript


async def extract_video_info_async(cleaned_url: str) -> Dict[str, Any]:
    """
    Async version of extract_video_info with better error handling.

    Args:
        cleaned_url: Validated YouTube URL

    Returns:
        Dictionary containing video metadata

    Raises:
        HTTPException: For various failure modes
    """
    if not os.getenv("APIFY_API_KEY"):
        raise create_error_response(500, ERROR_MESSAGES["apify_not_configured"])

    try:
        # Run in thread pool to avoid blocking
        scrapper_result = await run_in_executor(scrap_youtube, cleaned_url)

        return {
            "url": cleaned_url,
            "title": scrapper_result.title,
            "author": scrapper_result.channel.title if scrapper_result.channel else "Unknown Author",
            "duration": scrapper_result.durationFormatted,
            "thumbnail": scrapper_result.thumbnail,
            "view_count": scrapper_result.viewCountInt,
            "like_count": scrapper_result.likeCountInt,
            "upload_date": scrapper_result.publishDateText,
        }
    except Exception as e:
        error_msg = f"API extraction failed: {str(e)}"
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            raise create_error_response(429, ERROR_MESSAGES["api_quota_exceeded"])
        else:
            raise create_error_response(500, error_msg)


def create_transcript_from_analysis(analysis: Analysis) -> str:
    """
    Create transcript-like content from Gemini analysis.

    Args:
        analysis: Gemini analysis result

    Returns:
        Formatted transcript string
    """
    transcript_parts = [f"Video Analysis: {analysis.title}", "", "Summary:", analysis.summary, "", "Key Points:"]

    for chapter in analysis.chapters:
        transcript_parts.extend([f"\n{chapter.header}:", chapter.summary])

    return "\n".join(transcript_parts)


def format_summary_from_analysis(analysis: Analysis) -> str:
    """
    Format structured analysis into markdown summary.

    Args:
        analysis: Gemini analysis result

    Returns:
        Formatted markdown summary
    """
    summary_parts = [f"**{analysis.title}**", "", "**Summary:**", analysis.summary, "", "**Key Takeaways:**"]
    summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])

    if analysis.key_facts:
        summary_parts.extend(["", "**Key Facts:**"])
        summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])

    if analysis.chapters:
        summary_parts.extend(["", "**Chapters:**"])
        for chapter in analysis.chapters:
            summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

    return "\n".join(summary_parts)


# Legacy helper function - kept for backward compatibility in other endpoints
def convert_analysis_to_dict(analysis: Analysis) -> Dict[str, Any]:
    """
    Convert analysis result to dictionary format.

    NOTE: This is kept for backward compatibility in non-workflow endpoints.
    The /api/summarize endpoint now returns the Analysis object directly.
    """
    return {
        "title": analysis.title,
        "summary": analysis.summary,
        "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters],
        "key_facts": analysis.key_facts,
        "takeaways": analysis.takeaways,
        "keywords": analysis.keywords,
    }


def handle_remote_protocol_error(e: RemoteProtocolError, context: str = "processing") -> HTTPException:
    """
    Handle RemoteProtocolError with appropriate response.

    Args:
        e: RemoteProtocolError exception
        context: Context where error occurred

    Returns:
        Appropriate HTTPException
    """
    if "Server disconnected without sending a response" in str(e):
        return create_error_response(413, ERROR_MESSAGES["video_too_long"])
    else:
        return create_error_response(500, f"{ERROR_MESSAGES['processing_failed']}. Last error: {str(e)}")


async def extract_transcript_with_fallback_async(cleaned_url: str, logs: List[str]) -> Tuple[str, str, str, str]:
    """
    Async version of extract transcript using multi-tier fallback approach.

    Args:
        cleaned_url: Validated YouTube URL
        logs: Log accumulator list

    Returns:
        Tuple of (title, author, transcript, processing_method)

    Raises:
        HTTPException: If all methods fail
    """
    title = "Unknown Title"
    author = "Unknown Author"
    transcript = ""
    processing_method = "unknown"

    # Tier 1: Try Apify YouTube Scraper API
    if os.getenv("APIFY_API_KEY"):
        try:
            log_and_print("🔄 Tier 1: Trying Apify YouTube Scraper API...")
            logs.append("🔄 Tier 1: Trying Apify YouTube Scraper API...")

            # Run in thread pool to avoid blocking
            scrapper_result = await run_in_executor(scrap_youtube, cleaned_url)
            title, author, transcript = parse_video_content(scrapper_result)

            if transcript and transcript.strip() and not transcript.startswith("["):
                log_and_print("✅ Tier 1 successful: Got transcript from Apify API")
                logs.append("✅ Tier 1 successful: Got transcript from Apify API")
                processing_method = "apify_api"
                return title, author, transcript, processing_method
            else:
                log_and_print("❌ Tier 1 failed: No valid transcript from Apify API")
                logs.append("❌ Tier 1 failed: No valid transcript from Apify API")

        except Exception as e:
            error_msg = f"❌ Tier 1 failed: {str(e)}"
            log_and_print(error_msg)
            logs.append(error_msg)

            if "quota" in str(e).lower() or "limit" in str(e).lower():
                logs.append("⚠️ API quota/rate limit detected, proceeding to Tier 2...")

    else:
        log_and_print("⚠️ Tier 1 skipped: APIFY_API_KEY not configured")
        logs.append("⚠️ Tier 1 skipped: APIFY_API_KEY not configured")

        # Tier 2: Fall-back to Gemini direct URL processing
        if not os.getenv("GEMINI_API_KEY"):
            error_msg = "Apify API failed and GEMINI_API_KEY not configured"
            log_and_print(f"❌ {error_msg}")
            logs.append(f"❌ {error_msg}")
            raise create_error_response(500, error_msg)

        log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
        logs.append("🤖 Tier 2: Using Gemini direct URL processing...")

        try:
            # Run in thread pool to avoid blocking
            analysis = await run_in_executor(summarize_video, cleaned_url)

            title = analysis.title or "Unknown Title"
            author = "Unknown Author"  # Gemini doesn't provide author from URL
            transcript = create_transcript_from_analysis(analysis)
            processing_method = "gemini_direct"

            log_and_print("✅ Tier 2 successful: Got content from Gemini direct processing")
            logs.append("✅ Tier 2 successful: Got content from Gemini direct processing")

            return title, author, transcript, processing_method

        except RemoteProtocolError as e:
            raise handle_remote_protocol_error(e, "transcript extraction")
        except Exception as e:
            error_msg = f"❌ Tier 2 failed: {str(e)}"
            log_and_print(error_msg)
            logs.append(error_msg)
            raise create_error_response(500, f"{ERROR_MESSAGES['processing_failed']}. Last error: {str(e)}")


async def run_in_executor(func, *args):
    """Helper function to run CPU-bound operations in thread pool."""
    return await asyncio.get_event_loop().run_in_executor(None, func, *args)


async def generate_summary_from_content_async(content: str, content_type: str = "transcript") -> Tuple[str, Dict[str, Any]]:
    """
    Async version of generate summary and analysis from content.

    Args:
        content: Content to analyze (transcript or URL)
        content_type: Type of content ("transcript" or "url")

    Returns:
        Tuple of (formatted_summary, analysis_dict)

    Raises:
        Various exceptions for different failure modes
    """
    # Run in thread pool to avoid blocking
    analysis = await run_in_executor(summarize_video, content)
    summary = format_summary_from_analysis(analysis)
    analysis_dict = convert_analysis_to_dict(analysis)

    return summary, analysis_dict


# ================================
# ENHANCED API ENDPOINTS
# ================================


@app.get("/")
async def root():
    """Root endpoint with comprehensive API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/api/health",
        "optimizations": [
            "Async processing",
            "Response caching",
            "Rate limiting",
            "Enhanced error handling",
            "Request logging",
            "Performance monitoring",
        ],
        "endpoints": {
            "validate_url": "/api/validate-url",
            "video_info": "/api/video-info",
            "transcript": "/api/transcript",
            "summary": "/api/summary",
            "process": "/api/process",
            "generate": "/api/generate (Master API - orchestrates all capabilities)",
            "scrap": "/api/scrap (Extract video metadata and transcript)",
            "summarize": "/api/summarize (🔧 UPGRADED: Full LangGraph workflow with quality assessment)",
            "stream_process": "/api/stream-process (Streaming video processing with progress)",
            "summarize_stream": "/api/summarize-stream (LangGraph streaming with real-time progress)",
        },
        "workflow": {
            "tier_1": "Apify YouTube Scraper API",
            "tier_2": "Gemini direct URL processing",
        },
    }


# ================================
# API ROUTER ENDPOINTS (Essential endpoints with /api prefix)
# ================================


@api_router.get("/health")
async def api_health_check():
    """Health check endpoint with enhanced system status."""
    return {
        "status": "healthy",
        "message": f"{API_TITLE} is running",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
        "environment": {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "apify_configured": bool(os.getenv("APIFY_API_KEY")), "fal_configured": bool(os.getenv("FAL_KEY"))},
        "performance": {"cache_size": len(cache._cache), "uptime": "Available in production"},
    }


@api_router.post("/scrap", response_model=ScrapResponse)
@cache_response(ttl=1800)  # Cache for 30 minutes
async def scrap_video(request: ScrapRequest):
    """
    Scrap video info and transcript using Apify YouTube Scraper API.

    This endpoint directly uses the scrap_youtube() function to extract:
    - Video metadata (title, author, duration, views, etc.)
    - Full transcript with chapter formatting
    """
    start_time = datetime.now()

    try:
        cleaned_url = validate_url(request.url)
        log_and_print(f"🔄 Scraping video: {cleaned_url}")

        if not os.getenv("APIFY_API_KEY"):
            raise create_error_response(500, ERROR_MESSAGES["apify_not_configured"])

        # Use existing scrap_youtube function
        scrapper_result = await run_in_executor(scrap_youtube, cleaned_url)

        # Extract video info
        video_info = VideoInfoResponse(
            url=cleaned_url,
            title=scrapper_result.title,
            author=scrapper_result.channel.title if scrapper_result.channel else "Unknown Author",
            duration=scrapper_result.durationFormatted,
            thumbnail=scrapper_result.thumbnail,
            view_count=scrapper_result.viewCountInt,
            like_count=scrapper_result.likeCountInt,
            upload_date=scrapper_result.publishDateText,
        )

        # Extract transcript using existing parser
        transcript = scrapper_result.parsed_transcript

        processing_time = datetime.now() - start_time

        log_and_print(f"✅ Video scraped successfully: {video_info.title}")

        return ScrapResponse(status="success", message="Video scraped successfully", cleaned_url=cleaned_url, video_info=video_info, transcript=transcript, processing_time=f"{processing_time.total_seconds():.1f}s")

    except HTTPException:
        raise
    except Exception as e:
        processing_time = datetime.now() - start_time
        error_msg = f"Scraping failed: {str(e)}"
        log_and_print(f"❌ {error_msg}")

        if "quota" in str(e).lower() or "limit" in str(e).lower():
            raise create_error_response(429, ERROR_MESSAGES["api_quota_exceeded"])
        else:
            raise create_error_response(500, error_msg)


@api_router.post("/summarize", response_model=SummarizeResponse)
async def summarize_content(request: SummarizeRequest):
    """
    Generate comprehensive AI analysis using the full LangGraph workflow.

    This endpoint uses the complete LangGraph workflow to provide:
    - Structured chapters with key points
    - Key facts and takeaways with timestamps
    - Overall summary
    - Quality assessment with scoring details
    - Processing metadata (iterations, refinement cycles)

    Content can be either a YouTube URL or transcript text.
    The workflow includes automatic quality checking and refinement iterations.
    """
    start_time = datetime.now()

    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise create_error_response(500, ERROR_MESSAGES["gemini_not_configured"])

        log_and_print(f"🔧 Running full LangGraph workflow for {request.content_type}")
        log_and_print(f"📝 Content length: {len(request.content)} characters")
        log_and_print(f"📝 Content preview: {request.content[:200]}...")

        # Use the LangGraph workflow directly to get complete result
        def run_workflow():
            from youtube_summarizer.summarizer import (
                WorkflowInput,
                create_compiled_graph,
            )

            graph = create_compiled_graph()
            result = graph.invoke(WorkflowInput(transcript_or_url=request.content))
            return WorkflowOutput.model_validate(result)

        workflow_result = await run_in_executor(run_workflow)

        processing_time = datetime.now() - start_time

        log_and_print(f"✅ Analysis completed with {workflow_result.quality.percentage_score}% quality after {workflow_result.iteration_count} iterations")

        return SummarizeResponse(status="success", message="Analysis completed successfully", analysis=workflow_result.analysis, quality=workflow_result.quality, processing_time=f"{processing_time.total_seconds():.1f}s", iteration_count=workflow_result.iteration_count)

    except HTTPException:
        raise
    except RemoteProtocolError as e:
        processing_time = datetime.now() - start_time
        if "Server disconnected without sending a response" in str(e):
            error_msg = ERROR_MESSAGES["video_too_long"]
        else:
            error_msg = f"Analysis failed: {str(e)}"

        log_and_print(f"❌ {error_msg}")
        raise create_error_response(413, error_msg)
    except Exception as e:
        processing_time = datetime.now() - start_time
        error_msg = f"Full workflow analysis failed: {str(e)}"
        log_and_print(f"❌ {error_msg}")

        raise create_error_response(500, error_msg)


@api_router.post("/stream-process")
async def stream_process_video(request: YouTubeProcessRequest):
    """
    Stream complete video processing with real-time updates.

    This endpoint combines transcript extraction and analysis with streaming:
    - Real-time progress for transcript extraction
    - Streaming analysis with partial results
    - Quality checks and refinement iterations
    - Complete final result
    """
    start_time = datetime.now()
    cleaned_url = validate_url(request.url)
    logs = []

    async def generate_stream():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'step': 'start', 'message': 'Starting video processing...', 'url': cleaned_url, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Step 1: Extract transcript
            yield f"data: {json.dumps({'type': 'progress', 'step': 'transcript', 'message': 'Extracting transcript...', 'timestamp': datetime.now().isoformat()})}\n\n"

            title, author, transcript, processing_method = await extract_transcript_with_fallback_async(cleaned_url, logs)

            yield f"data: {json.dumps({'type': 'transcript_complete', 'step': 'transcript', 'data': {'title': title, 'author': author, 'method': processing_method, 'transcript_length': len(transcript)}, 'timestamp': datetime.now().isoformat()})}\n\n"

            if not request.generate_summary:
                # Just return transcript
                result = {"video_info": {"title": title, "author": author, "url": cleaned_url}, "transcript": transcript, "processing_method": processing_method, "logs": logs}
                yield f"data: {json.dumps({'type': 'final_result', 'step': 'complete', 'data': result, 'timestamp': datetime.now().isoformat()})}\n\n"
                return

            # Step 2: Stream the analysis
            yield f"data: {json.dumps({'type': 'progress', 'step': 'analysis', 'message': 'Starting AI analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"

            # Stream the LangGraph analysis
            analysis_result = None
            for chunk in stream_summarize_video(transcript):
                # Convert Pydantic model to dict and add timestamp
                chunk_dict = chunk.model_dump()
                chunk_dict["timestamp"] = datetime.now().isoformat()
                chunk_dict["type"] = "analysis_progress"

                yield f"data: {json.dumps(chunk_dict)}\n\n"

                # Capture final result
                if chunk.is_complete and chunk.analysis:
                    analysis_result = chunk_dict.get("analysis")

                await asyncio.sleep(0.1)

            # Send complete result
            processing_time = datetime.now() - start_time
            complete_result = {"video_info": {"title": title, "author": author, "url": cleaned_url}, "transcript": transcript, "analysis": analysis_result, "processing_method": processing_method, "processing_time": f"{processing_time.total_seconds():.1f}s", "logs": logs}

            yield f"data: {json.dumps({'type': 'complete', 'step': 'done', 'data': complete_result, 'timestamp': datetime.now().isoformat()})}\n\n"

            log_and_print(f"✅ Streaming video processing completed in {processing_time.total_seconds():.1f}s")

        except Exception as e:
            error_msg = f"Streaming processing failed: {str(e)}"
            log_and_print(f"❌ {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'logs': logs, 'timestamp': datetime.now().isoformat()})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@api_router.post("/summarize-stream")
async def stream_summarize_content(request: StreamSummarizeRequest):
    """
    Stream AI analysis using LangGraph with real-time progress updates.

    This endpoint uses the verified LangGraph streaming approach to provide:
    - Real-time progress updates for each workflow step
    - Partial results as they become available
    - Quality checks and refinement iterations
    - Final complete analysis with full graph state

    The streaming prevents timeouts while providing comprehensive results.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise create_error_response(500, ERROR_MESSAGES["gemini_not_configured"])

    log_and_print(f"🌊 Starting LangGraph streaming analysis for {request.content_type}")
    log_and_print(f"📝 Content length: {len(request.content)} characters")
    log_and_print(f"📝 Content preview: {request.content[:200]}...")

    async def generate_langgraph_stream():
        start_time = datetime.now()

        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting LangGraph analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"

            # Stream the LangGraph workflow with progress updates
            chunk_count = 0
            final_result = None

            for chunk in stream_summarize_video(request.content):
                chunk_count += 1

                # Convert Pydantic model to dict for JSON serialization
                chunk_dict = chunk.model_dump()
                chunk_dict["chunk_number"] = chunk_count
                chunk_dict["timestamp"] = datetime.now().isoformat()

                # Add progress information
                if chunk.quality:
                    chunk_dict["progress"] = {"iteration": chunk.iteration_count, "quality_score": chunk.quality.percentage_score, "is_acceptable": chunk.quality.is_acceptable, "total_score": f"{chunk.quality.total_score}/{chunk.quality.max_possible_score}"}
                else:
                    chunk_dict["progress"] = {"iteration": chunk.iteration_count, "status": "analysis_in_progress"}

                # Add step identification
                if chunk.analysis and not chunk.quality:
                    chunk_dict["step"] = "initial_analysis"
                elif chunk.analysis and chunk.quality and not chunk.is_complete:
                    chunk_dict["step"] = "quality_check"
                elif chunk.is_complete:
                    chunk_dict["step"] = "final_result"
                    final_result = chunk_dict

                # Send the chunk
                yield f"data: {json.dumps(chunk_dict)}\n\n"

                # Small delay for streaming visibility
                await asyncio.sleep(0.1)

            # Send completion summary
            processing_time = datetime.now() - start_time
            completion_data = {"type": "complete", "message": "LangGraph analysis completed successfully", "processing_time": f"{processing_time.total_seconds():.1f}s", "total_chunks": chunk_count, "timestamp": datetime.now().isoformat()}

            if final_result:
                completion_data["final_analysis"] = {"title": final_result.get("analysis", {}).get("title"), "quality_score": final_result.get("quality", {}).get("percentage_score"), "chapters_count": len(final_result.get("analysis", {}).get("chapters", [])), "is_acceptable": final_result.get("quality", {}).get("is_acceptable")}

            yield f"data: {json.dumps(completion_data)}\n\n"

            log_and_print(f"✅ LangGraph streaming analysis completed in {processing_time.total_seconds():.1f}s")
            log_and_print(f"📊 Total chunks processed: {chunk_count}")

        except Exception as e:
            error_msg = f"LangGraph streaming analysis failed: {str(e)}"
            log_and_print(f"❌ {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'timestamp': datetime.now().isoformat()})}\n\n"

    return StreamingResponse(
        generate_langgraph_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


# ================================
# INCLUDE API ROUTER (After all endpoints are defined)
# ================================

# Include the API router in the main app - this MUST come after all @api_router endpoints
app.include_router(api_router)


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    log_and_print(f"🚀 Starting {API_TITLE} v{API_VERSION} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
