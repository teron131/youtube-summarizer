"""
YouTube Summarizer FastAPI Application
=====================================

This FastAPI application provides a comprehensive API for YouTube video processing,
including downloading, transcription, and AI-powered summarization.

## 🔄 Processing Workflow

The application uses a multi-tier fallback approach for optimal results:

### Tier 1: yt-dlp Captions
- Attempts to extract existing captions/subtitles from YouTube
- Fastest method when captions are available
- Supports multiple languages (English, Chinese)

### Tier 2: Audio Transcription
- Downloads audio using yt-dlp when no captions exist
- Transcribes using FAL API with Whisper
- Handles videos without captions

### Tier 3: Gemini Direct Processing
- Directly processes YouTube URLs using Google's Gemini AI
- Fallback when other methods fail
- Provides analysis even for problematic videos

## 📊 API Endpoints

- `/api/validate-url` - Validate YouTube URLs
- `/api/video-info` - Extract video metadata
- `/api/transcript` - Get video transcripts
- `/api/summary` - Generate text summaries
- `/api/process` - Complete processing pipeline
- `/api/generate` - Master endpoint with all capabilities

## 🔧 Configuration

Set environment variables:
- `GEMINI_API_KEY` - For AI summarization
- `FAL_KEY` - For audio transcription
- `PORT` - Server port (default: 8080)
- `HOST` - Server host (default: 0.0.0.0)

Backend-only package for programmatic use and frontend integration.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from httpx import RemoteProtocolError
from pydantic import BaseModel, Field
from youtube_summarizer.summarizer import (
    clean_youtube_url,
    is_youtube_url,
    summarize_video,
)
from youtube_summarizer.utils import log_and_print
from youtube_summarizer.youtube_loader import extract_video_info, youtube_loader

load_dotenv()

# ================================
# CONSTANTS & CONFIGURATION
# ================================

API_VERSION = "2.0.0"
API_TITLE = "YouTube Summarizer API"
API_DESCRIPTION = "YouTube video processing with transcription & summarization"

# Error messages
ERROR_MESSAGES = {
    "invalid_url": "Invalid YouTube URL format",
    "empty_url": "YouTube URL is required",
    "gemini_not_configured": "GEMINI_API_KEY not configured",
    "fal_not_configured": "FAL_KEY not configured",
    "video_too_long": "Video is too long for processing. Please try with a shorter video or use time segments.",
    "processing_failed": "All processing methods failed",
}

# Processing status indicators
PROCESSING_STATUS = {
    "pending": "pending",
    "success": "success",
    "failed": "failed",
}

# ================================
# LOGGING CONFIGURATION
# ================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# PYDANTIC MODELS
# ================================
# All models kept in this file for frontend reference


class YouTubeRequest(BaseModel):
    """Basic YouTube URL request model."""

    url: str = Field(..., description="YouTube video URL", min_length=1)


class YouTubeProcessRequest(BaseModel):
    """Extended request model for processing with options."""

    url: str = Field(..., description="YouTube video URL", min_length=1)
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class TextSummaryRequest(BaseModel):
    """Request model for text-only summarization."""

    text: str = Field(..., description="Text content to summarize", min_length=1)


class GenerateRequest(BaseModel):
    """Comprehensive request model for the master generate endpoint."""

    url: str = Field(..., description="YouTube video URL", min_length=1)
    include_transcript: bool = Field(default=True, description="Include full transcript")
    include_summary: bool = Field(default=True, description="Include AI summary")
    include_analysis: bool = Field(default=True, description="Include structured analysis")
    include_metadata: bool = Field(default=True, description="Include video metadata")


class URLValidationResponse(BaseModel):
    """Response model for URL validation."""

    is_valid: bool = Field(description="Whether the URL is a valid YouTube URL")
    cleaned_url: Optional[str] = Field(description="Cleaned URL if valid")
    original_url: str = Field(description="Original URL provided")


class VideoInfoResponse(BaseModel):
    """Response model for video metadata."""

    title: str
    author: str
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    view_count: Optional[int] = None
    upload_date: Optional[str] = None
    url: str = Field(description="Cleaned YouTube URL")


class TranscriptResponse(BaseModel):
    """Response model for transcript extraction."""

    title: str
    author: str
    transcript: str
    url: str = Field(description="Cleaned YouTube URL")
    processing_time: str


class SummaryResponse(BaseModel):
    """Response model for text summarization."""

    title: str
    summary: str
    analysis: Optional[Dict[str, Any]] = None
    processing_time: str


class ProcessingResponse(BaseModel):
    """Response model for complete video processing."""

    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


class GenerateResponse(BaseModel):
    """Comprehensive response model for the master generate endpoint."""

    status: str
    message: str
    video_info: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    processing_details: Dict[str, str] = {}
    logs: List[str] = []


# ================================
# HELPER FUNCTIONS
# ================================


def validate_url(url: str) -> str:
    """
    Validate and clean YouTube URL.

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


def parse_video_content(video_content: str) -> Tuple[str, str, str]:
    """
    Parse video content from youtube_loader output.

    Args:
        video_content: Raw output from youtube_loader

    Returns:
        Tuple of (title, author, transcript)
    """
    title = "Unknown Title"
    author = "Unknown Author"
    transcript = ""

    lines = video_content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Title: "):
            title = line[7:]
        elif line.startswith("Author: "):
            author = line[8:]
        elif line.strip() == "subtitle:":
            transcript = "\n".join(lines[i + 1 :])
            break

    return title, author, transcript


def create_transcript_from_analysis(analysis) -> str:
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


def format_summary_from_analysis(analysis) -> str:
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


def convert_analysis_to_dict(analysis) -> Dict[str, Any]:
    """
    Convert analysis result to dictionary format.

    Args:
        analysis: Gemini analysis result

    Returns:
        Dictionary representation of analysis
    """
    return {"chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters], "key_facts": analysis.key_facts, "takeaways": analysis.takeaways, "overall_summary": analysis.overall_summary}


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
        return HTTPException(status_code=413, detail=ERROR_MESSAGES["video_too_long"])
    else:
        return HTTPException(status_code=500, detail=f"{ERROR_MESSAGES['processing_failed']}. Last error: {str(e)}")


def extract_transcript_with_fallback(cleaned_url: str, logs: List[str]) -> Tuple[str, str, str, str]:
    """
    Extract transcript using multi-tier fallback approach.

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

    # Tier 1: Try yt-dlp loader
    try:
        log_and_print("🔄 Tier 1: Trying yt-dlp loader...")
        logs.append("🔄 Tier 1: Trying yt-dlp loader...")

        video_content = youtube_loader(cleaned_url)
        title, author, transcript = parse_video_content(video_content)

        if transcript and transcript.strip() and not transcript.startswith("["):
            log_and_print("✅ Tier 1 successful: Got transcript from yt-dlp loader")
            logs.append("✅ Tier 1 successful: Got transcript from yt-dlp loader")
            processing_method = "yt-dlp_loader"
            return title, author, transcript, processing_method
        else:
            log_and_print("❌ Tier 1 failed: No valid transcript from yt-dlp loader")
            logs.append("❌ Tier 1 failed: No valid transcript from yt-dlp loader")

    except Exception as e:
        error_msg = f"❌ Tier 1 failed: {str(e)}"
        log_and_print(error_msg)
        logs.append(error_msg)

    # Tier 2: Fallback to Gemini direct URL processing
    if not os.getenv("GEMINI_API_KEY"):
        error_msg = "yt-dlp loader failed and GEMINI_API_KEY not configured"
        log_and_print(f"❌ {error_msg}")
        logs.append(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
    logs.append("🤖 Tier 2: Using Gemini direct URL processing...")

    try:
        analysis = summarize_video(cleaned_url)

        title = analysis.title if analysis.title else "Unknown Title"
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
        raise HTTPException(status_code=500, detail=f"{ERROR_MESSAGES['processing_failed']}. Last error: {str(e)}")


def generate_summary_from_content(content: str, content_type: str = "transcript") -> Tuple[str, Dict[str, Any]]:
    """
    Generate summary and analysis from content.

    Args:
        content: Content to analyze (transcript or URL)
        content_type: Type of content ("transcript" or "url")

    Returns:
        Tuple of (formatted_summary, analysis_dict)

    Raises:
        Various exceptions for different failure modes
    """
    analysis = summarize_video(content)
    summary = format_summary_from_analysis(analysis)
    analysis_dict = convert_analysis_to_dict(analysis)

    return summary, analysis_dict


# ================================
# API ENDPOINTS
# ================================


@app.get("/")
async def root():
    """Root endpoint with comprehensive API information."""
    return {"name": API_TITLE, "version": API_VERSION, "description": API_DESCRIPTION, "docs": "/docs", "health": "/api/health", "endpoints": {"validate_url": "/api/validate-url", "video_info": "/api/video-info", "transcript": "/api/transcript", "summary": "/api/summary", "process": "/api/process", "generate": "/api/generate (Master API - orchestrates all capabilities)"}, "workflow": {"tier_1": "yt-dlp captions extraction", "tier_2": "audio transcription via FAL", "tier_3": "Gemini direct URL processing"}}


@app.get("/api/health")
async def health_check():
    """Health check endpoint with system status."""
    return {"status": "healthy", "message": f"{API_TITLE} is running", "timestamp": datetime.now().isoformat(), "version": API_VERSION, "environment": {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "fal_configured": bool(os.getenv("FAL_KEY"))}}


@app.post("/api/validate-url", response_model=URLValidationResponse)
async def validate_youtube_url(request: YouTubeRequest):
    """Validate and clean YouTube URL."""
    try:
        is_valid = is_youtube_url(request.url)
        cleaned_url = clean_youtube_url(request.url) if is_valid else None

        return URLValidationResponse(is_valid=is_valid, cleaned_url=cleaned_url, original_url=request.url)
    except Exception as e:
        log_and_print(f"❌ URL validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL validation failed: {str(e)}")


@app.post("/api/video-info", response_model=VideoInfoResponse)
async def get_video_info(request: YouTubeRequest):
    """Extract basic video information without processing."""
    try:
        cleaned_url = validate_url(request.url)
        log_and_print(f"📋 Extracting video info for: {cleaned_url}")

        metadata = extract_video_info(cleaned_url)
        return VideoInfoResponse(**metadata)

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Video info extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract video info: {str(e)}")


@app.post("/api/transcript", response_model=TranscriptResponse)
async def get_video_transcript(request: YouTubeRequest):
    """Extract video transcript with multi-tier fallback approach."""
    start_time = datetime.now()

    try:
        cleaned_url = validate_url(request.url)
        log_and_print(f"📋 Extracting transcript for: {cleaned_url}")

        logs = []
        title, author, transcript, _ = extract_transcript_with_fallback(cleaned_url, logs)

        processing_time = datetime.now() - start_time
        return TranscriptResponse(title=title, author=author, transcript=transcript, url=cleaned_url, processing_time=f"{processing_time.total_seconds():.1f}s")

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
            raise HTTPException(status_code=500, detail=ERROR_MESSAGES["gemini_not_configured"])

        log_and_print("📋 Generating summary from provided text...")

        summary, analysis_dict = generate_summary_from_content(request.text, "text")
        processing_time = datetime.now() - start_time

        # Extract title from analysis
        analysis = summarize_video(request.text)

        return SummaryResponse(title=analysis.title, summary=summary, analysis=analysis_dict, processing_time=f"{processing_time.total_seconds():.1f}s")

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.post("/api/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeProcessRequest):
    """
    Complete YouTube video processing with multi-tier fallback approach.

    Workflow:
    1. URL validation and cleaning
    2. Tier 1: yt-dlp captions extraction
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
        title, author, transcript, processing_method = extract_transcript_with_fallback(cleaned_url, logs)

        # Step 3: Generate summary if requested
        summary = None
        analysis_data = None

        if request.generate_summary and transcript and not transcript.startswith("["):
            try:
                if processing_method == "gemini_direct":
                    log_and_print("📋 Using existing Gemini analysis for summary...")
                    logs.append("📋 Using existing Gemini analysis for summary...")
                    summary, analysis_data = generate_summary_from_content(cleaned_url, "url")
                else:
                    log_and_print("📋 Generating summary from transcript...")
                    logs.append("📋 Generating summary from transcript...")

                    if not os.getenv("GEMINI_API_KEY"):
                        summary = f"[{ERROR_MESSAGES['gemini_not_configured']}]"
                    else:
                        full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{transcript}"
                        summary, analysis_data = generate_summary_from_content(full_content, "transcript")

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
            "title": title,
            "author": author,
            "transcript": transcript,
            "summary": summary,
            "analysis": analysis_data,
            "processing_method": processing_method,
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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_comprehensive_analysis(request: GenerateRequest):
    """
    Master API endpoint that orchestrates all video processing capabilities.

    This is the most comprehensive endpoint providing:
    - URL validation and cleaning
    - Video metadata extraction
    - Multi-tier transcript extraction (yt-dlp + Gemini fallback)
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
    processing_details = {"url_validation": PROCESSING_STATUS["pending"], "metadata_extraction": PROCESSING_STATUS["pending"], "transcript_extraction": PROCESSING_STATUS["pending"], "summary_generation": PROCESSING_STATUS["pending"]}
    metadata = {"total_processing_time": "0s", "start_time": start_time.isoformat(), "api_version": API_VERSION}

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
                video_info = extract_video_info(cleaned_url)
                processing_details["metadata_extraction"] = PROCESSING_STATUS["success"]

                log_and_print(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")
                logs.append(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")

            except Exception as e:
                processing_details["metadata_extraction"] = f"failed: {str(e)}"
                error_msg = f"Metadata extraction failed: {str(e)}"
                log_and_print(f"⚠️ {error_msg}")
                logs.append(f"⚠️ {error_msg}")

                # Don't fail entire request for metadata issues
                video_info = {"title": "Unknown Title", "author": "Unknown Author", "url": cleaned_url, "error": str(e)}

        # Step 3: Transcript Extraction
        if request.include_transcript:
            log_and_print("📝 Step 3: Extracting transcript with multi-tier approach...")
            logs.append("📝 Step 3: Extracting transcript with multi-tier approach...")

            try:
                title, author, transcript_content, method = extract_transcript_with_fallback(cleaned_url, logs)
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
                            summary, _ = generate_summary_from_content(content_to_analyze)

                        if request.include_analysis:
                            analysis_result = summarize_video(content_to_analyze)
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

        return GenerateResponse(status="success", message="Comprehensive video analysis completed successfully", video_info=video_info, transcript=transcript, summary=summary, analysis=analysis, metadata=metadata, processing_details=processing_details, logs=logs)

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
• Zelenskyy said that rearming and strengthening Ukraine's military will be part of any security guarantees.

**Chapters:**
**Introduction and Welcome**
The video begins with a live news report from the White House, where the press is being led into the Oval Office. President Donald Trump is meeting with Ukrainian President Volodymyr Zelenskyy.

**Zelenskyy's Remarks and a Letter to the First Lady**
President Zelenskyy thanks President Trump for the invitation and for his personal efforts to stop the killings and the war.
"""

    # Example analysis structure
    example_analysis = {
        "title": "Trump Holds Meeting with Zelensky in the Oval Office",
        "overall_summary": "In a significant diplomatic meeting at the White House, President Donald Trump hosted Ukrainian President Volodymyr Zelenskyy in the Oval Office to discuss the ongoing war with Russia. Trump expressed optimism about making substantial progress towards peace, highlighting his recent discussions with Russia's president and upcoming talks with European leaders. A key announcement from the meeting was Trump's intention to call Vladimir Putin immediately following the discussions, with the potential for a trilateral summit between the three leaders to broker a peace deal.",
        "chapters": [
            {"header": "Introduction and Welcome", "summary": "The video begins with a live news report from the White House, where the press is being led into the Oval Office. President Donald Trump is meeting with Ukrainian President Volodymyr Zelenskyy. Trump starts by welcoming Zelenskyy, stating it's an honor to have him. He mentions they've had good discussions and that substantial progress is being made. He also refers to a recent meeting with the President of Russia and a forthcoming meeting with seven European leaders, highlighting the importance of the current discussions.", "key_points": ["Donald Trump welcomes Ukrainian President Volodymyr Zelenskyy to the Oval Office.", "Trump states that substantial progress is being made in their discussions.", "He mentions a recent good meeting with the President of Russia and an upcoming meeting with seven powerful European leaders."]},
            {"header": "Zelenskyy's Remarks and a Letter to the First Lady", "summary": "President Zelenskyy thanks President Trump for the invitation and for his personal efforts to stop the killings and the war. He also takes the opportunity to thank the First Lady of the United States for sending a letter to Vladimir Putin concerning abducted Ukrainian children. Zelenskyy then hands Trump a letter from his own wife, the First Lady of Ukraine, addressed to Trump's wife, which Trump accepts with a laugh.", "key_points": ["Zelenskyy thanks Trump for the invitation and his personal efforts to stop the war.", "Hethanks the First Lady of the United States for sending a letter to Putin about abducted Ukrainian children.", "Zelenskyy presents a letter from his wife to Trump's wife."]},
            {"header": "Press Questions on Ending the War", "summary": "The press begins to ask questions. A reporter points out the differing perspectives, with Zelenskyy stating Russia must end the war it started, and Trump suggesting Zelenskyy could end it almost immediately. Trump responds by saying he believes a trilateral meeting between himself, Zelenskyy, and Putin could be arranged if the current discussions are successful. He asserts that this trilateral meeting would have a reasonable chance of ending the war.", "key_points": ["A reporter questions the differing views on who should end the war, citing statements from both leaders.", "Trump expresses confidence in the possibility of a trilateral meeting with Zelenskyy and Putin if the day's meetings go well.", "Trump believes there is a reasonable chance of ending the war through such a meeting."]},
            {"header": "US Support and Security Guarantees", "summary": 'A reporter asks if this meeting is a "deal or no deal" moment for American support to Ukraine. Trump dismisses the idea that it\'s the "end of the road," stating the priority is to stop the ongoing killing. When asked about the security guarantees he needs, Zelenskyy says it involves everything, specifically mentioning the need for a strong Ukrainian army, weapons, training, and intelligence. When asked if security guarantees could involve U.S. troops, Trump does not rule it out, stating, "We\'ll be involved" and that European leaders also want to provide protection.', "key_points": ["Trump is asked if this meeting represents the end of the road for American support for Ukraine.", "Trump denies it's the end of the road, emphasizing the goal is to stop the killing.", "Zelenskyy states that security guarantees would involve strengthening and rearming the Ukrainian military.", "Trump does not rule out sending U.S. troops to Ukraine as part of a security arrangement."]},
        ],
        "key_facts": ["Donald Trump held a meeting with Ukrainian President Volodymyr Zelenskyy in the Oval Office.", 'Trump stated he would have a trilateral meeting with Zelenskyy and Putin "if everything works out well today."', 'Trump announced he would telephone Vladimir Putin "right after" his meeting with Zelenskyy.', "Zelenskyy said that rearming and strengthening Ukraine's military will be part of any security guarantees.", "Trump did not rule out sending U.S. troops to Ukraine to ensure security as part of a peace deal.", "Zelenskyy delivered a letter from his wife to Melania Trump concerning abducted Ukrainian children.", "Trump claimed that Putin wants the war on Ukraine to end."],
        "takeaways": ["Donald Trump is actively positioning himself as a central figure in negotiating an end to the war in Ukraine.", "A potential trilateral summit between the US, Ukraine, and Russia is being floated as a path to peace.", "The nature of future security guarantees for Ukraine, potentially involving US and European forces, is a critical point of negotiation.", "Despite past tensions, the meeting between Trump and Zelenskyy appeared cordial, signaling a potential shift in their dynamic.", "Ukraine's strategy for peace involves not just a cessation of hostilities but also significant military strengthening and concrete security guarantees from international partners."],
        "chapter_count": 4,
        "total_key_facts": 7,
        "total_takeaways": 5,
    }

    # Example logs
    example_logs = ["🎭 Generating example response for demonstration purposes...", "✅ This is a demonstration of the YouTube Summarizer's comprehensive analysis capabilities.", "🔍 In real usage, the system would validate the YouTube URL and extract video metadata.", "📝 The multi-tier transcript extraction would attempt yt-dlp captions, then audio transcription, then Gemini direct processing.", "🤖 AI analysis would generate structured insights including chapters, key facts, and takeaways.", "🎉 Example response generated successfully!"]

    # Example processing details
    example_processing_details = {"url_validation": "example_mode", "metadata_extraction": "example_mode", "transcript_extraction": "example_mode", "summary_generation": "example_mode"}

    # Example metadata
    processing_time = datetime.now() - start_time
    example_metadata = {"total_processing_time": f"{processing_time.total_seconds():.1f}s", "start_time": start_time.isoformat(), "end_time": datetime.now().isoformat(), "api_version": API_VERSION, "mode": "example_demonstration", "original_url": "example", "cleaned_url": "https://youtube.com/watch?v=example", "steps_completed": 4, "steps_total": 4}

    log_and_print("🎉 Example response generated successfully!")

    return GenerateResponse(status="success", message="Example demonstration of comprehensive video analysis capabilities", video_info=example_video_info, transcript=example_transcript, summary=example_summary, analysis=example_analysis, metadata=example_metadata, processing_details=example_processing_details, logs=example_logs)


# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    log_and_print(f"🚀 Starting {API_TITLE} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
