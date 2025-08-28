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
allowed_origins = os.getenv("ALLOWED_ORIGINS", f"http://localhost:3000,http://localhost:8080,{os.getenv('VITE_API_BASE_URL')}").split(",")
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
        return VideoInfoResponse(**metadata)

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
    example_video_info = {
        "url": "https://www.youtube.com/watch?v=A5w-dEgIU1M",
        "title": "The Trillion Dollar Equation",
        "author": "Veritasium",
        "duration": "00:31:22",
        "thumbnail": "https://img.youtube.com/vi/A5w-dEgIU1M/maxresdefault.jpg",
        "view_count": 13462116,
        "like_count": 321234,
        "upload_date": "Feb 27, 2024",
    }

    # Example transcript
    example_transcript = """- This single equation spawned four multi-trillion dollar industries and transformed everyone's approach to risk. Do you think that most people are aware of the size, scale, utility of derivatives? - No. No idea. - But at its core, this equation comes from physics, from discovering atoms, understanding how heat is transferred, and how to beat the casino at blackjack. So maybe it shouldn't be surprising that some of the best to beat the stock market were not veteran traders, but physicists, scientists, and mathematicians. In 1988, a mathematics professor named Jim Simons set up the Medallion Investment Fund, and every year for the next 30 years, the Medallion fund delivered higher returns than the market average, and not just by a little bit, it returned 66% per year. At that rate of growth, $100 invested in 1988 would be worth $8.4 billion today. This made Jim Simons easily the richest mathematician of all time. But being good at math doesn't guarantee success in financial markets. Just ask Isaac Newton. In 1720 Newton was 77 years old, and he was rich. He had made a lot of money working as a professor at Cambridge for decades, and he had a side hustle as the Master of the Royal Mint. His net worth was £30,000 the equivalent of $6 million today. Now, to grow his fortune, Newton invested in stocks. One of his big bets was on the South Sea Company. Their business was shipping enslaved Africans across the Atlantic. Business was booming and the share price grew rapidly. By April of 1720, the value of Newton's shares had doubled. So he sold his stock. But the stock price kept going up and by June, Newton bought back in and he kept buying shares even as the price peaked. When the price started to fall, Newton didn't sell. He bought more shares thinking he was buying the dip. But there was no rebound, and ultimately he lost around a third of his wealth. When asked why he didn't see it coming, Newton responded, \"I can calculate the motions of the heavenly bodies, but not the madness of people.\" So what did Simons get right that Newton got wrong? Well, for one thing, Simons was able to stand on the shoulders of giants. The pioneer of using math to model financial markets was Louis Bachelier, born in 1870. Both of his parents died when he was 18 and he had to take over his father's wine business. He sold the business a few years later and moved to Paris to study physics, but he needed a job to support himself and his family and he found one at the Bourse, The Paris Stock Exchange. And inside was Newton's \"madness of people\" in its rawest form. Hundreds of traders screaming prices, making hand signals, and doing deals. The thing that captured Bachelier's interest were contracts known as options. The earliest known options were bought around 600 BC by the Greek philosopher Thales of Miletus. He believed that the coming summer would yield a bumper crop of olives. To make money off this idea, he could have purchased olive presses, which if you were right, would be in great demand, but he didn't have enough money to buy the machines. So instead he went to all the existing olive press owners and paid them a little bit of money to secure the option to rent their presses in the summer for a specified price. When the harvest came, Thales was right, there were so many olives that the price of renting a press skyrocketed. Thales paid the press owners their pre-agreed price, and then he rented out the machines at a higher rate and pocketed the difference. Thales had executed the first known call option. A call option gives you the right, but not the obligation to buy something at a later date for a set price known as the strike price. You can also buy a put option, which gives you the right, but not the obligation to sell something at a later date for the strike price. Put options are useful if you expect the price to go down. Call options are useful if you expect the price to go up. For example, let's say the current price of Apple stock is a hundred dollars, but you expect it to go up. You could buy a call option for $10 that gives you the right, but not the obligation to buy Apple stock in one year for a hundred dollars. That is the strike price. Just a little side note, American options can be exercised on any date up to the expiry, whereas European options must be exercised on the expiry date. To keep things simple, we'll stick to European options. So if in a year the price of Apple stock has gone up to $130, you can use the option to buy shares for a hundred dollars and then immediately sell them for $130. After you take into account the $10 you paid for the option, you've made a $20 profit. Alternatively, if in a year the stock prices dropped to $70, you just wouldn't use the option and you've lost the $10 you paid for it. So the profit and loss diagram looks like this. If the stock price ends up below the strike price, you lose what you paid for the option. But if the stock price is higher than the strike price, then you earn that difference minus the cost of the option. There are at least three advantages of options. One is that it limits your downside. If you had bought the stock instead of the option and it went down to $70, you would've lost $30. And in theory, you could have lost a hundred if the stock went to zero. The second benefit is options provide leverage. If you had bought the stock and it went up to $130, then your investment grew by 30%. But if you had bought the option, you only had to put up $10. So your profit of $20 is actually a 200% return on investment. On the downside, if you had owned the stock, your investment would've only dropped by 30%, whereas with the option you lose all 100%. So with options trading, there's a chance to make much larger profits, but also much bigger losses. The third benefit is you can use options as a hedge. - I think the original motivation for options was to figure out a way to reduce risk. And then of course, once people decided they wanted to buy insurance, that meant that there are other people out there that wanted to sell it or a profit, and that's how markets get created. - So options can be an incredibly useful investing tool, but what Bachelier saw on the trading floor was chaos, especially when it came to the price of stock options. Even though they had been around for hundreds of years, no one had found a good way to price them. Traders would just bargain to come to an agreement about what the price should be. - Given the option to buy or sell something in the future, it seems like a very amorphous kind of a trade. And so coming up with prices for these rather strange objects has been a challenge that's plagued a number of economists and business people for centuries. - Now, Bachelier, already interested in probability, thought there had to be a mathematical solution to this problem, and he proposed this as his PhD topic to his advisor Henri Poincaré. Looking into the math of finance wasn't really something people did back then, but to Bachelier's surprise, Poincaré agreed. To accurately price an option, first you need to know what happens to stock prices over time. The price of a stock is basically set by a tug of war between buyers and sellers. When more people wanna buy a stock, the price goes up. When more people wanna sell a stock, the price goes down. But the number of buyers and sellers can be influenced by almost anything, like the weather, politics, new competitors, innovation and so on. So Bachelier realized that it's virtually impossible to predict all these factors accurately. So the best you can do is assume that at any point in time the stock price is just as likely to go up as down and therefore over the long term, stock prices follow a random walk, moving up and down as if their next move is determined by the flip of a coin. - Randomness is a hallmark of an efficient market. By efficient economists typically"""

    # Example summary
    example_summary = """**The Trillion Dollar Equation**

**Overall Summary:**
The video explores the profound impact of a single mathematical concept—the random walk—on the world of finance, leading to the creation of multi-trillion dollar industries. It traces the history of quantitative finance from Isaac Newton's failed speculation to the pioneering work of Louis Bachelier, who first modeled stock prices as a random process, unknowingly predating Einstein's similar work on Brownian motion. The narrative highlights key innovators like Ed Thorp, who applied blackjack strategies to develop dynamic hedging, and culminates with the 1973 Black-Scholes-Merton equation. This formula provided a definitive method for pricing options, unleashing an explosion in derivatives trading. These instruments are shown to be double-edged swords, used by corporations for hedging risk and by traders for high-stakes leverage, as exemplified by the GameStop saga. Finally, the video examines the success of quantitative hedge funds like Jim Simons' Medallion Fund, which challenge the 'Efficient Market Hypothesis' by using advanced mathematics and machine learning to find and exploit hidden market patterns. The ultimate irony presented is that the work of these 'quants' in exploiting market inefficiencies simultaneously helps to eliminate them, pushing the financial world ever closer to the perfectly random model it was first imagined to be.

**Key Takeaways:**
• Mathematical models, many originating from physics, have fundamentally transformed modern finance by providing a framework to price risk and complex financial instruments.
• Options are versatile financial tools that allow investors to either hedge against potential losses or use leverage to amplify potential gains, but with correspondingly higher risk.
• The Black-Scholes-Merton equation provided a standardized, universally accepted formula for pricing options, which catalyzed the creation of multi-trillion dollar derivative markets.
• While financial markets are largely efficient and random, they are not perfectly so. Sophisticated quantitative analysis can uncover subtle patterns and inefficiencies, allowing skilled investors to consistently 'beat the market'.
• The very act of exploiting market inefficiencies with mathematical models helps to eliminate those same inefficiencies, ironically pushing the market closer to a state of perfect randomness.
• Derivatives have a dual nature: they can enhance market stability and liquidity during normal times but can also amplify risk and exacerbate crashes during periods of market stress.

**Key Facts:**
• Jim Simons' Medallion Investment Fund returned 66% per year for 30 years.
• A $100 investment in the Medallion Fund in 1988 would be worth $8.4 billion today.
• In 1720, Isaac Newton lost approximately one-third of his wealth in the South Sea Company stock bubble.
• The earliest known use of options was by Thales of Miletus around 600 BC.
• Louis Bachelier proposed his mathematical theory of speculation for his PhD in 1900, five years before Einstein's paper on Brownian motion.
• Ed Thorp's hedge fund achieved a 20% annual return for 20 consecutive years.
• The Black-Scholes-Merton equation was published in 1973.
• The global derivatives market is estimated to be on the order of several hundred trillion dollars.
• During the 2021 short squeeze, GameStop shares rose approximately 700%.
• Myron Scholes and Robert Merton were awarded the Nobel Prize in Economics in 1997 for their work on option pricing.

**Chapters:**
**Introduction: Quants vs. The Madness of People**
The video introduces the immense impact of a single mathematical equation on the financial world. It contrasts the extraordinary success of quantitative investors like mathematician Jim Simons with the historical failure of brilliant minds like Isaac Newton, who was undone by market psychology. This sets up the central theme: the quest to use mathematics to model and profit from the seemingly chaotic and unpredictable nature of financial markets.

**The Origins of Options and Bachelier's Insight**
This chapter explains the fundamental concept of financial options, tracing their origin back to ancient Greece. It defines call and put options using a modern example of Apple stock, highlighting their three main advantages: limited downside, leverage, and hedging. The narrative then introduces Louis Bachelier, who, while working at the Paris Stock Exchange, pioneered the use of mathematics to model financial markets, specifically focusing on the chaotic problem of how to price options.

**The Random Walk and Brownian Motion**
This section delves into Louis Bachelier's core idea: that stock prices follow a random walk. Using the analogy of a Galton board, it shows how numerous individual random paths create a predictable collective pattern—a normal distribution. This mathematical framework, which Bachelier called the 'radiation of probabilities', was a rediscovery of the heat equation. The chapter highlights a significant historical parallel, as Albert Einstein independently developed the same mathematics five years later to explain Brownian motion, thereby proving the existence of atoms.

**Bachelier's Pricing Model and Its Limitations**
The chapter explains how Bachelier used his random walk model to create a formula for pricing options. By calculating the probabilities of different price outcomes, he determined that a 'fair price' is one that equalizes the expected profit or loss for both the buyer and the seller. Although he had beaten Einstein to the mathematics of the random walk and solved a long-standing financial puzzle, his groundbreaking thesis went unnoticed for decades.

**From Blackjack to Wall Street: Ed Thorp's Innovations**
This chapter introduces Ed Thorp, who successfully transitioned from being a professional blackjack player (inventing card counting) to a highly successful hedge fund manager. He applied his understanding of odds and risk management to the stock market, pioneering the concept of dynamic hedging to create risk-neutral positions. Thorp also refined Bachelier's random walk model by adding a 'drift' component to account for underlying trends in stock prices, creating a more accurate option pricing formula which he used privately for his fund.

**The Black-Scholes-Merton Equation**
The video reaches its central topic: the Black-Scholes-Merton equation. In 1973, these economists developed a new and robust method for pricing options. Their crucial assumption was that a risk-free portfolio, constructed through dynamic hedging, must yield the same return as the safest available asset. This principle allowed them to derive a precise mathematical formula that relates an option's price to variables like the stock price, time, and interest rates. The equation was a revelation and was rapidly adopted by the financial industry, transforming it forever.

**The Impact and Scale of Derivatives**
This chapter explores the profound consequences of the Black-Scholes-Merton equation. It fueled the explosive growth of derivatives markets, now valued in the hundreds of trillions of dollars. The video explains the dual nature of these instruments: they allow corporations to hedge real-world risks, but they also provide immense leverage, as demonstrated by the GameStop phenomenon. While derivatives can enhance market stability by distributing risk, they also have the potential to amplify systemic shocks and contribute to major financial crises.

**Beating the Market: Jim Simons and Renaissance Technologies**
The narrative returns to Jim Simons, who represents the pinnacle of quantitative investing. After a distinguished career in mathematics, Simons applied his expertise in pattern recognition and code-breaking to financial markets. By hiring top scientists and leveraging computational power and vast amounts of data, his firm Renaissance Technologies sought to find non-random patterns that others missed. Their flagship Medallion Fund achieved unprecedented returns, serving as a powerful counterexample to the idea that the market is perfectly efficient and unbeatable.

**Conclusion: The Quest for an Efficient Market**
The video concludes by synthesizing its main themes. The work of physicists and mathematicians has provided deep insights into market dynamics, risk, and pricing. While the Efficient Market Hypothesis is a powerful model, evidence from funds like Medallion shows that inefficiencies and predictable patterns can be found and exploited. In a final, ironic twist, the very success of these quantitative strategies helps to correct the market's imperfections. As quants profit from patterns, they eliminate them, pushing the market ever closer to the ideal state of perfect, unpredictable randomness.
"""

    # Example analysis structure (condensed for readability)
    example_analysis = {
        "title": "The Trillion Dollar Equation",
        "overall_summary": "The video explores the profound impact of a single mathematical concept—the random walk—on the world of finance, leading to the creation of the 'trillion-dollar equation'. It traces the intellectual history from Louis Bachelier's pioneering application of random walk theory to stock prices in 1900, a concept that predated Einstein's use of it to explain Brownian motion, through the practical innovations of Ed Thorp, who took strategies from the blackjack table to Wall Street. The narrative culminates with the 1973 Black-Scholes-Merton equation, which provided a revolutionary formula for pricing financial options. This breakthrough unlocked the modern derivatives market, now a several-hundred-trillion-dollar industry used by companies and investors globally for hedging and leverage. The video contrasts the theory of an efficient, random market with the real-world success of quantitative investors like Jim Simons, whose Medallion Fund used advanced mathematics and machine learning to consistently beat the market for decades. Ultimately, it reveals a central irony of modern finance: the very act of using science to find predictive patterns in the market helps to eliminate those patterns, pushing the financial world ever closer to a state of perfect, unpredictable randomness.",
        "chapters": [
            {
                "header": "Introduction: The Quants vs. The Market",
                "summary": "The video introduces the immense impact of a single mathematical equation on the financial world. It contrasts the staggering success of quantitative investor Jim Simons, whose Medallion Fund generated 66% annual returns for three decades, with the historical failure of Sir Isaac Newton in the stock market. This sets the stage to explore what Simons and other 'quants' understood that Newton did not, pointing to the power of mathematical models over human intuition.",
                "key_points": [
                    "A single equation from physics spawned multi-trillion dollar industries and transformed risk management.",
                    "Mathematician Jim Simons' Medallion Fund achieved an unprecedented 66% annual return for 30 years, turning $100 into $8.4 billion.",
                    "In contrast, physicist Isaac Newton lost a significant portion of his fortune in the South Sea Company bubble, highlighting that intelligence alone isn't enough to beat the market's 'madness'.",
                ],
            },
            {
                "header": "The Origins and Mechanics of Options",
                "summary": "This chapter explains the fundamental concept of financial options. It traces their origin back to ancient Greece with the story of Thales and the olive presses. The summary details the mechanics of call options (betting on a price increase) and put options (betting on a price decrease), using a modern example with Apple stock. It highlights the key benefits of using options: limiting potential losses, achieving leverage to amplify returns, and serving as a form of insurance or 'hedging' to reduce risk.",
                "key_points": [
                    "Options are financial contracts giving the right, but not the obligation, to buy (call option) or sell (put option) an asset at a predetermined price by a future date.",
                    "The earliest known use of options was by Greek philosopher Thales of Miletus around 600 BC to speculate on an olive harvest.",
                    "Options offer three main advantages: limiting downside risk, providing leverage for potentially higher returns (and losses), and hedging against price fluctuations.",
                ],
            },
        ],
        "key_facts": [
            "Jim Simons' Medallion Fund delivered an average return of 66% per year for 30 years.",
            "In 1720, Isaac Newton lost approximately one-third of his wealth in the South Sea Company stock bubble.",
        ],
        "takeaways": [
            "Mathematical models originating from physics have revolutionized finance, enabling the creation of multi-trillion dollar industries and transforming how risk is managed.",
            "Options are powerful financial instruments that allow for leverage, risk limitation, and hedging, but their complexity requires sophisticated pricing models.",
        ],
        "chapter_count": 2,
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
        "cleaned_url": "https://www.youtube.com/watch?v=A5w-dEgIU1M",
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
