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
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from httpx import RemoteProtocolError
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

from youtube_summarizer.summarizer import Analysis, summarize_video
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url, log_and_print
from youtube_summarizer.youtube_scrapper import (
    YouTubeScrapperResult,
    parse_transcript,
    scrap_youtube,
)

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

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# Rate limiting
app.add_middleware(RateLimitMiddleware, requests_per_minute=RATE_LIMIT_REQUESTS)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# CORS with more restrictive defaults (can be overridden via env vars)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
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
    if any(path in str(request.url) for path in ["/process", "/generate", "/summary"]):
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

    @validator("url")
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

    @validator("url")
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


class VideoInfoResponse(BaseResponse):
    """Response model for video metadata."""

    url: str = Field(description="Cleaned YouTube URL")
    title: str
    author: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
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

    video_info: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    processing_details: Dict[str, str] = {}
    logs: List[str] = []


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
    transcript = parse_transcript(scrapper_result)

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
        scrapper_result = await asyncio.get_event_loop().run_in_executor(None, scrap_youtube, cleaned_url)

        return {
            "url": cleaned_url,
            "title": scrapper_result.title,
            "author": scrapper_result.channel.title if scrapper_result.channel else "Unknown Author",
            "duration": scrapper_result.durationFormatted,
            "thumbnail": scrapper_result.thumbnail,
            "view_count": scrapper_result.viewCountInt,
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
    transcript_parts = [f"Video Analysis: {analysis.title}", "", "Overall Summary:", analysis.overall_summary, "", "Key Points:"]

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
    summary_parts = [f"**{analysis.title}**", "", "**Overall Summary:**", analysis.overall_summary, "", "**Key Takeaways:**"]
    summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])

    if analysis.key_facts:
        summary_parts.extend(["", "**Key Facts:**"])
        summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])

    if analysis.chapters:
        summary_parts.extend(["", "**Chapters:**"])
        for chapter in analysis.chapters:
            summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

    return "\n".join(summary_parts)


def convert_analysis_to_dict(analysis: Analysis) -> Dict[str, Any]:
    """
    Convert analysis result to dictionary format.

    Args:
        analysis: Gemini analysis result

    Returns:
        Dictionary representation of analysis
    """
    return {
        "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters],
        "key_facts": analysis.key_facts,
        "takeaways": analysis.takeaways,
        "overall_summary": analysis.overall_summary,
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
            scrapper_result = await asyncio.get_event_loop().run_in_executor(None, scrap_youtube, cleaned_url)
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

    # Tier 2: Fallback to Gemini direct URL processing
    if not os.getenv("GEMINI_API_KEY"):
        error_msg = "Apify API failed and GEMINI_API_KEY not configured"
        log_and_print(f"❌ {error_msg}")
        logs.append(f"❌ {error_msg}")
        raise create_error_response(500, error_msg)

    log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
    logs.append("🤖 Tier 2: Using Gemini direct URL processing...")

    try:
        # Run in thread pool to avoid blocking
        analysis = await asyncio.get_event_loop().run_in_executor(None, summarize_video, cleaned_url)

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
    analysis = await asyncio.get_event_loop().run_in_executor(None, summarize_video, content)
    summary = format_summary_from_analysis(analysis)
    analysis_dict = convert_analysis_to_dict(analysis)

    return summary, analysis_dict


# ================================
# BACKWARDS COMPATIBILITY FUNCTIONS
# ================================


def extract_video_info(cleaned_url: str) -> Dict[str, Any]:
    """
    Backwards-compatible synchronous wrapper for extract_video_info_async.

    This function exists for test compatibility and legacy usage.
    For new code, use extract_video_info_async directly.

    Args:
        cleaned_url: Validated YouTube URL

    Returns:
        Dictionary containing video metadata

    Raises:
        HTTPException: For various failure modes
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(extract_video_info_async(cleaned_url))


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
        "health": "/health",
        "optimizations": ["Async processing", "Response caching", "Rate limiting", "Enhanced error handling", "Request logging", "Performance monitoring"],
        "endpoints": {"validate_url": "/validate-url", "video_info": "/video-info", "transcript": "/transcript", "summary": "/summary", "process": "/process", "generate": "/generate (Master API - orchestrates all capabilities)"},
        "workflow": {"tier_1": "Apify YouTube Scraper API", "tier_2": "Gemini direct URL processing"},
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with enhanced system status."""
    return {"status": "healthy", "message": f"{API_TITLE} is running", "timestamp": datetime.now().isoformat(), "version": API_VERSION, "environment": {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "apify_configured": bool(os.getenv("APIFY_API_KEY")), "fal_configured": bool(os.getenv("FAL_KEY"))}, "performance": {"cache_size": len(cache._cache), "uptime": "Available in production"}}


@app.post("/validate-url", response_model=URLValidationResponse)
@cache_response(ttl=3600)  # Cache for 1 hour
async def validate_youtube_url(request: YouTubeRequest):
    """Validate and clean YouTube URL with caching."""
    try:
        is_valid = is_youtube_url(request.url)
        cleaned_url = clean_youtube_url(request.url) if is_valid else None

        return URLValidationResponse(
            status="success",
            message="URL validation completed",
            is_valid=is_valid,
            cleaned_url=cleaned_url,
            original_url=request.url,
        )
    except Exception as e:
        log_and_print(f"❌ URL validation failed: {str(e)}")
        return URLValidationResponse(
            status="error",
            message=f"URL validation failed: {str(e)}",
            is_valid=False,
            cleaned_url=None,
            original_url=request.url,
        )


@app.post("/video-info", response_model=VideoInfoResponse)
@cache_response(ttl=1800)  # Cache for 30 minutes
async def get_video_info(request: YouTubeRequest):
    """Extract basic video information without processing."""
    try:
        cleaned_url = validate_url(request.url)
        log_and_print(f"📋 Extracting video info for: {cleaned_url}")

        metadata = await extract_video_info_async(cleaned_url)
        return VideoInfoResponse(status="success", message="Video information extracted successfully", **metadata)

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Video info extraction failed: {str(e)}")
        raise create_error_response(400, f"Failed to extract video info: {str(e)}")


@app.post("/transcript", response_model=TranscriptResponse)
@cache_response(ttl=3600)  # Cache for 1 hour
async def get_video_transcript(request: YouTubeRequest):
    """Extract video transcript with multi-tier fallback approach."""
    start_time = datetime.now()

    try:
        cleaned_url = validate_url(request.url)
        log_and_print(f"📋 Extracting transcript for: {cleaned_url}")

        logs = []
        title, author, transcript, method = await extract_transcript_with_fallback_async(cleaned_url, logs)

        processing_time = datetime.now() - start_time
        return TranscriptResponse(
            status="success",
            message="Transcript extracted successfully",
            title=title,
            author=author,
            transcript=transcript,
            url=cleaned_url,
            processing_time=f"{processing_time.total_seconds():.1f}s",
            source=method,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Transcript extraction failed: {str(e)}")
        raise create_error_response(400, f"Failed to extract transcript: {str(e)}")


@app.post("/summary", response_model=SummaryResponse)
async def generate_text_summary(request: TextSummaryRequest):
    """Generate summary from provided text content - OPTIMIZED to avoid duplicate API calls."""
    start_time = datetime.now()

    try:
        if not os.getenv("GEMINI_API_KEY"):
            raise create_error_response(500, ERROR_MESSAGES["gemini_not_configured"])

        log_and_print("📋 Generating summary from provided text...")

        # OPTIMIZATION: Single API call instead of duplicate calls
        analysis = await asyncio.get_event_loop().run_in_executor(None, summarize_video, request.text)

        # Format results from single analysis
        summary = format_summary_from_analysis(analysis)
        analysis_dict = convert_analysis_to_dict(analysis)
        processing_time = datetime.now() - start_time

        return SummaryResponse(
            status="success",
            message="Summary generated successfully",
            title=analysis.title,
            summary=summary,
            analysis=analysis_dict,
            processing_time=f"{processing_time.total_seconds():.1f}s",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Summary generation failed: {str(e)}")
        raise create_error_response(500, f"Failed to generate summary: {str(e)}")


@app.post("/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeProcessRequest):
    """
    Complete YouTube video processing with multi-tier fallback approach.

    Workflow:
    1. URL validation and cleaning
    2. Tier 1: Apify YouTube Scraper API
    3. Tier 2: Gemini direct URL processing
    4. Optional: AI summary generation
    """
    start_time = datetime.now()
    logs = [f"🎬 Starting processing: {request.url}"]

    try:
        # Step 1: URL validation
        cleaned_url = validate_url(request.url)
        log_and_print(f"🔗 Cleaned URL: {cleaned_url}")
        logs.append(f"🔗 Cleaned URL: {cleaned_url}")

        # Step 2: Extract transcript with fallback
        title, author, transcript, processing_method = await extract_transcript_with_fallback_async(cleaned_url, logs)

        # Step 3: Generate summary if requested
        summary = None
        analysis_data = None

        if request.generate_summary and transcript and not transcript.startswith("["):
            try:
                if processing_method == "gemini_direct":
                    log_and_print("📋 Using existing Gemini analysis for summary...")
                    logs.append("📋 Using existing Gemini analysis for summary...")
                    summary, analysis_data = await generate_summary_from_content_async(cleaned_url, "url")
                else:
                    log_and_print("📋 Generating summary from transcript...")
                    logs.append("📋 Generating summary from transcript...")

                    if not os.getenv("GEMINI_API_KEY"):
                        summary = f"[{ERROR_MESSAGES['gemini_not_configured']}]"
                    else:
                        full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{transcript}"
                        summary, analysis_data = await generate_summary_from_content_async(full_content, "transcript")

                log_and_print("✅ Summary generated successfully")
                logs.append("✅ Summary generated successfully")

            except RemoteProtocolError as e:
                if "Server disconnected without sending a response" in str(e):
                    summary = f"[{ERROR_MESSAGES['video_too_long']}]"
                else:
                    summary = f"[Summary generation failed: {str(e)}]"

                error_msg = f"❌ Summary generation failed: {str(e)}"
                log_and_print(error_msg)
                logs.append(error_msg)

            except Exception as e:
                summary = f"[Summary generation failed: {str(e)}]"
                error_msg = f"❌ Summary generation failed: {str(e)}"
                log_and_print(error_msg)
                logs.append(error_msg)

        # Final response
        processing_time = datetime.now() - start_time
        completion_msg = f"✅ Processing completed in {processing_time.total_seconds():.1f}s using {processing_method}"
        log_and_print(completion_msg)
        logs.append(completion_msg)

        result_data = {
            "url": cleaned_url,
            "original_url": request.url,
            "title": title,
            "author": author,
            "transcript": transcript,
            "summary": summary,
            "analysis": analysis_data,
            "processing_method": processing_method,
            "processing_time": f"{processing_time.total_seconds():.1f}s",
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


@app.post("/generate", response_model=GenerateResponse)
async def generate_comprehensive_analysis(request: GenerateRequest):
    """
    Master API endpoint that orchestrates all video processing capabilities.

    This is the most comprehensive endpoint providing:
    - URL validation and cleaning
    - Video metadata extraction
    - Multi-tier transcript extraction (Apify API + Gemini fallback)
    - AI-powered summarization and analysis
    - Structured data output with detailed logging

    Perfect for frontend applications requiring complete video analysis.
    """
    start_time = datetime.now()

    # Check for example request
    if not request.url.strip() or request.url.strip().lower() == "example":
        return await generate_example_response(start_time)

    logs = [f"🚀 Starting comprehensive analysis: {request.url}"]

    # Initialize response containers
    video_info = None
    transcript = None
    summary = None
    analysis = None
    processing_details = {
        "url_validation": PROCESSING_STATUS["pending"],
        "metadata_extraction": PROCESSING_STATUS["pending"],
        "transcript_extraction": PROCESSING_STATUS["pending"],
        "summary_generation": PROCESSING_STATUS["pending"],
    }
    metadata = {
        "total_processing_time": "0s",
        "start_time": start_time.isoformat(),
        "api_version": API_VERSION,
    }

    try:
        # Step 1: URL Validation
        log_and_print("🔍 Step 1: Validating and cleaning URL...")
        logs.append("🔍 Step 1: Validating and cleaning URL...")

        try:
            cleaned_url = validate_url(request.url)
            processing_details["url_validation"] = PROCESSING_STATUS["success"]

            log_and_print(f"✅ URL validated and cleaned: {cleaned_url}")
            logs.append(f"✅ URL validated and cleaned: {cleaned_url}")

            metadata.update({"original_url": request.url, "cleaned_url": cleaned_url})

        except HTTPException as e:
            processing_details["url_validation"] = f"failed: {e.detail}"
            logs.append(f"❌ {e.detail}")
            raise

        # Step 2: Video Metadata Extraction
        if request.include_metadata:
            log_and_print("📋 Step 2: Extracting video metadata...")
            logs.append("📋 Step 2: Extracting video metadata...")

            try:
                video_info = await extract_video_info_async(cleaned_url)
                processing_details["metadata_extraction"] = PROCESSING_STATUS["success"]

                log_and_print(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")
                logs.append(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")

            except Exception as e:
                processing_details["metadata_extraction"] = f"failed: {str(e)}"
                error_msg = f"Metadata extraction failed: {str(e)}"
                log_and_print(f"⚠️ {error_msg}")
                logs.append(f"⚠️ {error_msg}")

                # Don't fail entire request for metadata issues
                video_info = {
                    "title": "Unknown Title",
                    "author": "Unknown Author",
                    "url": cleaned_url,
                    "error": str(e),
                }

        # Step 3: Transcript Extraction
        if request.include_transcript:
            log_and_print("📝 Step 3: Extracting transcript with multi-tier approach...")
            logs.append("📝 Step 3: Extracting transcript with multi-tier approach...")

            try:
                title, author, transcript_content, method = await extract_transcript_with_fallback_async(cleaned_url, logs)
                processing_details["transcript_extraction"] = f"success ({method})"
                transcript = transcript_content

                # Update video info with extracted title/author if available
                if video_info and title != "Unknown Title":
                    video_info["title"] = title
                if video_info and author != "Unknown Author":
                    video_info["author"] = author

            except HTTPException as e:
                processing_details["transcript_extraction"] = f"failed: {e.detail}"
                transcript = f"[Transcript extraction failed: {e.detail}]"
                logs.append(f"❌ Transcript extraction failed: {e.detail}")
            except Exception as e:
                processing_details["transcript_extraction"] = f"failed: {str(e)}"
                transcript = f"[Transcript extraction failed: {str(e)}]"
                logs.append(f"❌ Transcript extraction failed: {str(e)}")

        # Step 4: Summary and Analysis Generation
        if request.include_summary or request.include_analysis:
            log_and_print("🤖 Step 4: Generating AI summary and analysis...")
            logs.append("🤖 Step 4: Generating AI summary and analysis...")

            try:
                if not os.getenv("GEMINI_API_KEY"):
                    error_msg = ERROR_MESSAGES["gemini_not_configured"]
                    processing_details["summary_generation"] = f"failed: {error_msg}"
                    summary = f"[Summary generation failed: {error_msg}]"
                    analysis = {"error": error_msg}
                else:
                    # Determine content to analyze
                    if transcript and not transcript.startswith("["):
                        log_and_print("📋 Using extracted transcript for analysis...")
                        logs.append("📋 Using extracted transcript for analysis...")

                        title_for_analysis = video_info.get("title", "Unknown") if video_info else "Unknown"
                        full_content = f"Title: {title_for_analysis}\nTranscript:\n{transcript}"
                        content_to_analyze = full_content
                    else:
                        log_and_print("🔗 Using direct URL for analysis...")
                        logs.append("🔗 Using direct URL for analysis...")
                        content_to_analyze = cleaned_url

                    try:
                        if request.include_summary:
                            summary, _ = await generate_summary_from_content_async(content_to_analyze)

                        if request.include_analysis:
                            analysis_result = await asyncio.get_event_loop().run_in_executor(None, summarize_video, content_to_analyze)
                            analysis = {"title": analysis_result.title, "overall_summary": analysis_result.overall_summary, "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis_result.chapters], "key_facts": analysis_result.key_facts, "takeaways": analysis_result.takeaways, "chapter_count": len(analysis_result.chapters), "total_key_facts": len(analysis_result.key_facts), "total_takeaways": len(analysis_result.takeaways)}

                        processing_details["summary_generation"] = PROCESSING_STATUS["success"]
                        log_and_print("✅ Summary and analysis generated successfully")
                        logs.append("✅ Summary and analysis generated successfully")

                    except RemoteProtocolError as e:
                        if "Server disconnected without sending a response" in str(e):
                            error_msg = ERROR_MESSAGES["video_too_long"]
                        else:
                            error_msg = f"Summary generation failed: {str(e)}"

                        processing_details["summary_generation"] = f"failed: {error_msg}"
                        summary = f"[{error_msg}]"
                        analysis = {"error": error_msg}

                        log_and_print(f"❌ {error_msg}")
                        logs.append(f"❌ {error_msg}")

            except Exception as e:
                processing_details["summary_generation"] = f"failed: {str(e)}"
                error_msg = f"Summary generation failed: {str(e)}"
                summary = f"[{error_msg}]"
                analysis = {"error": str(e)}

                log_and_print(f"❌ {error_msg}")
                logs.append(f"❌ {error_msg}")

        # Finalize response
        processing_time = datetime.now() - start_time
        metadata.update({"total_processing_time": f"{processing_time.total_seconds():.1f}s", "end_time": datetime.now().isoformat(), "steps_completed": sum(1 for status in processing_details.values() if status.startswith("success")), "steps_total": len(processing_details)})

        completion_msg = f"🎉 Comprehensive analysis completed in {processing_time.total_seconds():.1f}s"
        log_and_print(completion_msg)
        logs.append(completion_msg)

        return GenerateResponse(
            status="success",
            message="Comprehensive video analysis completed successfully",
            video_info=video_info,
            transcript=transcript,
            summary=summary,
            analysis=analysis,
            metadata=metadata,
            processing_details=processing_details,
            logs=logs,
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = datetime.now() - start_time
        error_message = f"Comprehensive analysis failed: {str(e)}"

        metadata.update({"total_processing_time": f"{processing_time.total_seconds():.1f}s", "end_time": datetime.now().isoformat(), "error": str(e)})

        log_and_print(f"❌ {error_message}")
        logs.append(f"❌ {error_message}")

        return GenerateResponse(status="error", message=error_message, metadata=metadata, processing_details=processing_details, logs=logs)


async def generate_example_response(start_time: datetime) -> GenerateResponse:
    """Generate example response for demonstration purposes."""

    log_and_print("🎭 Generating example response for demonstration...")

    # Example video info
    example_video_info = {"title": "Trump Holds Meeting with Zelensky in the Oval Office", "author": "CNN", "duration": "480s", "duration_seconds": 480, "thumbnail": "https://img.youtube.com/vi/example/hqdefault.jpg", "view_count": 125000, "upload_date": "20240115", "url": "https://youtube.com/watch?v=example"}

    # Example transcript
    example_transcript = """This is a demonstration transcript showing how the YouTube Summarizer processes video content.

In this example meeting between President Trump and President Zelensky, several key diplomatic discussions take place regarding the ongoing conflict in Ukraine and potential paths toward peace.

The meeting covers topics including security guarantees, potential trilateral negotiations with Russia, and the role of international partners in ensuring lasting peace agreements.

This transcript demonstrates the type of content that would be extracted from a real YouTube video and then analyzed by the AI system."""

    # Example summary
    example_summary = """**Trump Holds Meeting with Zelensky in the Oval Office**

**Overall Summary:**
In a significant diplomatic meeting at the White House, President Donald Trump hosted Ukrainian President Volodymyr Zelenskyy in the Oval Office to discuss the ongoing war with Russia. Trump expressed optimism about making substantial progress towards peace, highlighting his recent discussions with Russia's president and upcoming talks with European leaders.

**Key Takeaways:**
• Donald Trump is actively positioning himself as a central figure in negotiating an end to the war in Ukraine.
• A potential trilateral summit between the US, Ukraine, and Russia is being floated as a path to peace.
• The nature of future security guarantees for Ukraine, potentially involving US and European forces, is a critical point of negotiation.
• Despite past tensions, the meeting between Trump and Zelenskyy appeared cordial, signaling a potential shift in their dynamic.

**Key Facts:**
• Donald Trump held a meeting with Ukrainian President Volodymyr Zelenskyy in the Oval Office.
• Trump stated he would have a trilateral meeting with Zelenskyy and Putin "if everything works out well today."
• Trump announced he would telephone Vladimir Putin "right after" his meeting with Zelenskyy.
• Zelenskyy said that rearming and strengthening Ukraine's military would be part of any security guarantees.

**Chapters:**
**Introduction and Welcome**
The video begins with a live news report from the White House, where the press is being led into the Oval Office. President Donald Trump is meeting with Ukrainian President Volodymyr Zelenskyy.

**Zelenskyy's Remarks and a Letter to the First Lady**
President Zelenskyy thanks President Trump for the invitation and for his personal efforts to stop the killings and the war.
"""

    # Example analysis structure (condensed for readability)
    example_analysis = {
        "title": "Trump Holds Meeting with Zelensky in the Oval Office",
        "overall_summary": "In a significant diplomatic meeting at the White House, President Donald Trump hosted Ukrainian President Volodymyr Zelenskyy in the Oval Office to discuss the ongoing war with Russia.",
        "chapters": [{"header": "Introduction and Welcome", "summary": "The video begins with a live news report from the White House...", "key_points": ["Trump welcomes Zelenskyy", "Progress mentioned", "European meetings planned"]}],
        "key_facts": ["Meeting held in Oval Office", "Trilateral summit discussed"],
        "takeaways": ["Trump positioning as peace negotiator", "Potential summit proposed"],
        "chapterCount": 1,
        "total_key_facts": 2,
        "total_takeaways": 2,
    }

    # Example logs
    example_logs = [
        "🎭 Generating example response for demonstration purposes...",
        "✅ This is a demonstration of the YouTube Summarizer's comprehensive analysis capabilities.",
        "🔍 In real usage, the system would validate the YouTube URL and extract video metadata.",
        "📝 The multi-tier transcript extraction would attempt Apify API, then Gemini direct processing.",
        "🤖 AI analysis would generate structured insights including chapters, key facts, and takeaways.",
        "🎉 Example response generated successfully!",
    ]

    # Example processing details
    example_processing_details = {
        "url_validation": "example_mode",
        "metadata_extraction": "example_mode",
        "transcript_extraction": "example_mode",
        "summary_generation": "example_mode",
    }

    # Example metadata
    processing_time = datetime.now() - start_time
    example_metadata = {
        "total_processing_time": f"{processing_time.total_seconds():.1f}s",
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "api_version": API_VERSION,
        "mode": "example_demonstration",
        "original_url": "example",
        "cleaned_url": "https://youtube.com/watch?v=example",
        "steps_completed": 4,
        "steps_total": 4,
    }

    log_and_print("🎉 Example response generated successfully!")

    return GenerateResponse(
        status="success",
        message="Example demonstration of comprehensive video analysis capabilities",
        video_info=example_video_info,
        transcript=example_transcript,
        summary=example_summary,
        analysis=example_analysis,
        metadata=example_metadata,
        processing_details=example_processing_details,
        logs=example_logs,
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
