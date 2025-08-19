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

from youtube_summarizer.summarizer import (
    clean_youtube_url,
    is_youtube_url,
    summarize_video,
)
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
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for your use case
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add redirect for Railway health check compatibility
from fastapi.responses import RedirectResponse


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


class GenerateRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL", min_length=1)
    include_transcript: bool = Field(default=True, description="Include full transcript")
    include_summary: bool = Field(default=True, description="Include AI summary")
    include_analysis: bool = Field(default=True, description="Include structured analysis")
    include_metadata: bool = Field(default=True, description="Include video metadata")


class GenerateResponse(BaseModel):
    status: str
    message: str
    video_info: Optional[Dict[str, Any]] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    processing_details: Dict[str, str] = {}
    logs: List[str] = []


# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {"name": "YouTube Summarizer API", "version": "2.0.0", "description": "YouTube video processing with transcription & summarization", "docs": "/docs", "health": "/api/health", "endpoints": {"validate_url": "/api/validate-url", "video_info": "/api/video-info", "transcript": "/api/transcript", "summary": "/api/summary", "process": "/api/process", "generate": "/api/generate (Master API - orchestrates all capabilities)"}}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "YouTube Summarizer API is running", "timestamp": datetime.now().isoformat(), "version": "2.0.0"}


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
        # Validate and clean URL
        if not is_youtube_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

        cleaned_url = clean_youtube_url(request.url)
        log_and_print(f"📋 Extracting video info for: {cleaned_url}")

        # Use yt-dlp to get basic metadata
        import yt_dlp

        ydl_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(cleaned_url, download=False)

        metadata = {"title": info.get("title"), "author": info.get("uploader"), "duration": f"{info.get('duration', 0)}s", "thumbnail": info.get("thumbnail"), "view_count": info.get("view_count"), "upload_date": info.get("upload_date"), "url": cleaned_url}

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
        # Validate and clean URL
        if not is_youtube_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")

        cleaned_url = clean_youtube_url(request.url)
        log_and_print(f"📋 Extracting transcript for: {cleaned_url}")

        # Tier 1: Try yt-dlp loader
        title = "Unknown Title"
        author = "Unknown Author"
        transcript = ""

        try:
            log_and_print("🔄 Tier 1: Trying yt-dlp loader...")
            video_content = youtube_loader(cleaned_url)

            # Parse the content to extract metadata and transcript
            lines = video_content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Title: "):
                    title = line[7:]
                elif line.startswith("Author: "):
                    author = line[8:]
                elif line.strip() == "subtitle:":
                    transcript = "\n".join(lines[i + 1 :])
                    break

            if transcript and transcript.strip() and not transcript.startswith("["):
                log_and_print("✅ Tier 1 successful: Got transcript from yt-dlp loader")
                processing_time = datetime.now() - start_time
                return TranscriptResponse(title=title, author=author, transcript=transcript, url=cleaned_url, processing_time=f"{processing_time.total_seconds():.1f}s")
            else:
                log_and_print("❌ Tier 1 failed: No valid transcript from yt-dlp loader")

        except Exception as e:
            log_and_print(f"❌ Tier 1 failed: {str(e)}")

        # Tier 2: Fallback to Gemini direct URL processing
        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="yt-dlp loader failed and GEMINI_API_KEY not configured")

        log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
        try:
            analysis = summarize_video(cleaned_url)  # Pass URL directly to Gemini

            # Extract basic info and create transcript from analysis
            title = analysis.title if analysis.title else "Unknown Title"
            author = "Unknown Author"  # Gemini doesn't provide author from URL

            # Create transcript-like content from analysis
            transcript_parts = [f"Video Analysis: {analysis.title}", "", "Overall Summary:", analysis.overall_summary, "", "Key Points:"]

            for chapter in analysis.chapters:
                transcript_parts.extend([f"\n{chapter.header}:", chapter.summary])

            transcript = "\n".join(transcript_parts)

            log_and_print("✅ Tier 2 successful: Got content from Gemini direct processing")
            processing_time = datetime.now() - start_time
            return TranscriptResponse(title=title, author=author, transcript=transcript, url=cleaned_url, processing_time=f"{processing_time.total_seconds():.1f}s")

        except Exception as gemini_error:
            log_and_print(f"❌ Tier 2 failed: {str(gemini_error)}")
            raise HTTPException(status_code=500, detail=f"All processing methods failed. Last error: {str(gemini_error)}")

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
        summary_parts = [f"**{analysis.title}**", "", "**Overall Summary:**", analysis.overall_summary, "", "**Key Takeaways:**"]
        summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])

        if analysis.key_facts:
            summary_parts.extend(["", "**Key Facts:**"])
            summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])

        if analysis.chapters:
            summary_parts.extend(["", "**Chapters:**"])
            for chapter in analysis.chapters:
                summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

        summary = "\n".join(summary_parts)
        processing_time = datetime.now() - start_time

        return SummaryResponse(title=analysis.title, summary=summary, analysis={"chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters], "key_facts": analysis.key_facts, "takeaways": analysis.takeaways, "overall_summary": analysis.overall_summary}, processing_time=f"{processing_time.total_seconds():.1f}s")

    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.post("/api/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeProcessRequest):
    """
    Complete YouTube video processing with multi-tier fallback approach.

    Tier 1: yt-dlp (captions/transcription)
    Tier 2: Gemini direct URL processing
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

        # Tier 1: Try yt-dlp loader
        title = "Unknown Title"
        author = "Unknown Author"
        transcript = ""
        processing_method = "unknown"

        log_and_print("🔄 Tier 1: Trying yt-dlp loader...")
        logs.append("🔄 Tier 1: Trying yt-dlp loader...")

        try:
            video_content = youtube_loader(cleaned_url)

            # Parse the content to extract metadata and transcript
            lines = video_content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Title: "):
                    title = line[7:]
                elif line.startswith("Author: "):
                    author = line[8:]
                elif line.strip() == "subtitle:":
                    transcript = "\n".join(lines[i + 1 :])
                    break

            if transcript and transcript.strip() and not transcript.startswith("["):
                log_and_print("✅ Tier 1 successful: Got transcript from yt-dlp loader")
                logs.append("✅ Tier 1 successful: Got transcript from yt-dlp loader")
                processing_method = "yt-dlp_loader"
            else:
                log_and_print("❌ Tier 1 failed: No valid transcript from yt-dlp loader")
                logs.append("❌ Tier 1 failed: No valid transcript from yt-dlp loader")
                transcript = ""

        except Exception as loader_error:
            error_msg = f"❌ Tier 1 failed: {str(loader_error)}"
            log_and_print(error_msg)
            logs.append(error_msg)
            transcript = ""

        # Tier 2: Fallback to Gemini direct URL processing
        if not transcript:
            if not os.getenv("GEMINI_API_KEY"):
                error_msg = "yt-dlp loader failed and GEMINI_API_KEY not configured"
                log_and_print(f"❌ {error_msg}")
                logs.append(f"❌ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

            log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
            logs.append("🤖 Tier 2: Using Gemini direct URL processing...")

            try:
                analysis = summarize_video(cleaned_url)  # Pass URL directly to Gemini

                # Extract info from analysis
                title = analysis.title if analysis.title else "Unknown Title"
                author = "Unknown Author"  # Gemini doesn't provide author from URL

                # Create transcript-like content from analysis
                transcript_parts = [f"Video Analysis: {analysis.title}", "", "Overall Summary:", analysis.overall_summary, "", "Key Points:"]

                for chapter in analysis.chapters:
                    transcript_parts.extend([f"\n{chapter.header}:", chapter.summary])

                transcript = "\n".join(transcript_parts)
                processing_method = "gemini_direct"

                log_and_print("✅ Tier 2 successful: Got content from Gemini direct processing")
                logs.append("✅ Tier 2 successful: Got content from Gemini direct processing")

            except Exception as gemini_error:
                error_msg = f"❌ Tier 2 failed: {str(gemini_error)}"
                log_and_print(error_msg)
                logs.append(error_msg)
                raise HTTPException(status_code=500, detail=f"All processing methods failed. Last error: {str(gemini_error)}")

        # Generate summary if requested and transcript is available
        summary = None
        analysis_data = None

        if request.generate_summary and transcript and not transcript.startswith("["):
            if processing_method == "gemini_direct":
                # If we used Gemini for transcript, we already have analysis
                log_and_print("📋 Using existing Gemini analysis for summary...")
                logs.append("📋 Using existing Gemini analysis for summary...")

                # We already have the analysis from Gemini processing
                try:
                    analysis = summarize_video(cleaned_url)  # Get fresh analysis for summary

                    # Convert structured analysis to summary text
                    summary_parts = [f"**{analysis.title}**", "", "**Overall Summary:**", analysis.overall_summary, "", "**Key Takeaways:**"]
                    summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])

                    if analysis.key_facts:
                        summary_parts.extend(["", "**Key Facts:**"])
                        summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])

                    if analysis.chapters:
                        summary_parts.extend(["", "**Chapters:**"])
                        for chapter in analysis.chapters:
                            summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

                    summary = "\n".join(summary_parts)

                    # Store structured analysis data
                    analysis_data = {"chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters], "key_facts": analysis.key_facts, "takeaways": analysis.takeaways, "overall_summary": analysis.overall_summary}

                    log_and_print("✅ Summary generated from Gemini analysis")
                    logs.append("✅ Summary generated from Gemini analysis")

                except Exception as summary_error:
                    error_msg = f"❌ Summary generation failed: {str(summary_error)}"
                    log_and_print(error_msg)
                    logs.append(error_msg)
                    summary = f"[Summary generation failed: {str(summary_error)}]"

            else:
                # Use transcript from yt-dlp loader for summary generation
                log_and_print("📋 Generating summary from transcript...")
                logs.append("📋 Generating summary from transcript...")

                if not os.getenv("GEMINI_API_KEY"):
                    log_and_print("❌ GEMINI_API_KEY not configured")
                    logs.append("❌ GEMINI_API_KEY not configured")
                    summary = "[GEMINI_API_KEY not configured - please set your Gemini API key]"
                else:
                    try:
                        full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{transcript}"
                        analysis = summarize_video(full_content)

                        # Convert structured analysis to summary text
                        summary_parts = [f"**{analysis.title}**", "", "**Overall Summary:**", analysis.overall_summary, "", "**Key Takeaways:**"]
                        summary_parts.extend([f"• {takeaway}" for takeaway in analysis.takeaways])

                        if analysis.key_facts:
                            summary_parts.extend(["", "**Key Facts:**"])
                            summary_parts.extend([f"• {fact}" for fact in analysis.key_facts])

                        if analysis.chapters:
                            summary_parts.extend(["", "**Chapters:**"])
                            for chapter in analysis.chapters:
                                summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

                        summary = "\n".join(summary_parts)

                        # Store structured analysis data
                        analysis_data = {"chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis.chapters], "key_facts": analysis.key_facts, "takeaways": analysis.takeaways, "overall_summary": analysis.overall_summary}

                        log_and_print("✅ Summary generated from transcript")
                        logs.append("✅ Summary generated from transcript")
                    except Exception as summary_error:
                        error_msg = f"❌ Summary generation failed: {str(summary_error)}"
                        log_and_print(error_msg)
                        logs.append(error_msg)
                        summary = f"[Summary generation failed: {str(summary_error)}]"

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

    This endpoint provides a one-stop-shop for comprehensive YouTube video analysis:
    - URL validation and cleaning
    - Video metadata extraction
    - Multi-tier transcript extraction (yt-dlp + Gemini fallback)
    - AI-powered summarization and analysis
    - Structured data output with detailed logging
    """
    start_time = datetime.now()
    logs = [f"🚀 Starting comprehensive analysis: {request.url}"]

    # Initialize response data containers
    video_info = None
    transcript = None
    summary = None
    analysis = None
    processing_details = {"url_validation": "pending", "metadata_extraction": "pending", "transcript_extraction": "pending", "summary_generation": "pending"}
    metadata = {"total_processing_time": "0s", "start_time": start_time.isoformat(), "api_version": "2.0.0"}

    try:
        # Step 1: URL Validation and Cleaning
        log_and_print("🔍 Step 1: Validating and cleaning URL...")
        logs.append("🔍 Step 1: Validating and cleaning URL...")

        try:
            if not request.url.strip():
                raise ValueError("URL cannot be empty")

            if not is_youtube_url(request.url):
                raise ValueError("Invalid YouTube URL format")

            cleaned_url = clean_youtube_url(request.url)
            processing_details["url_validation"] = "success"

            log_and_print(f"✅ URL validated and cleaned: {cleaned_url}")
            logs.append(f"✅ URL validated and cleaned: {cleaned_url}")

            metadata.update({"original_url": request.url, "cleaned_url": cleaned_url})

        except Exception as e:
            processing_details["url_validation"] = f"failed: {str(e)}"
            error_msg = f"URL validation failed: {str(e)}"
            log_and_print(f"❌ {error_msg}")
            logs.append(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Step 2: Video Metadata Extraction (if requested)
        if request.include_metadata:
            log_and_print("📋 Step 2: Extracting video metadata...")
            logs.append("📋 Step 2: Extracting video metadata...")

            try:
                import yt_dlp

                ydl_opts = {"quiet": True, "no_warnings": True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(cleaned_url, download=False)

                video_info = {"title": info.get("title"), "author": info.get("uploader"), "duration": f"{info.get('duration', 0)}s", "duration_seconds": info.get("duration", 0), "thumbnail": info.get("thumbnail"), "view_count": info.get("view_count"), "upload_date": info.get("upload_date"), "url": cleaned_url}

                processing_details["metadata_extraction"] = "success"
                log_and_print(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")
                logs.append(f"✅ Metadata extracted: {video_info['title']} by {video_info['author']}")

            except Exception as e:
                processing_details["metadata_extraction"] = f"failed: {str(e)}"
                error_msg = f"Metadata extraction failed: {str(e)}"
                log_and_print(f"⚠️ {error_msg}")
                logs.append(f"⚠️ {error_msg}")
                # Don't fail the entire request for metadata issues
                video_info = {"title": "Unknown Title", "author": "Unknown Author", "url": cleaned_url, "error": str(e)}

        # Step 3: Transcript Extraction (if requested)
        if request.include_transcript:
            log_and_print("📝 Step 3: Extracting transcript with multi-tier approach...")
            logs.append("📝 Step 3: Extracting transcript with multi-tier approach...")

            try:
                # Use the existing multi-tier approach from /api/transcript
                title = "Unknown Title"
                author = "Unknown Author"
                transcript_content = ""

                # Tier 1: Try yt-dlp loader
                try:
                    log_and_print("🔄 Tier 1: Trying yt-dlp loader...")
                    logs.append("🔄 Tier 1: Trying yt-dlp loader...")

                    video_content = youtube_loader(cleaned_url)

                    # Parse the content to extract metadata and transcript
                    lines = video_content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("Title: "):
                            title = line[7:]
                        elif line.startswith("Author: "):
                            author = line[8:]
                        elif line.strip() == "subtitle:":
                            transcript_content = "\n".join(lines[i + 1 :])
                            break

                    if transcript_content and transcript_content.strip() and not transcript_content.startswith("["):
                        log_and_print("✅ Tier 1 successful: Got transcript from yt-dlp loader")
                        logs.append("✅ Tier 1 successful: Got transcript from yt-dlp loader")
                        processing_details["transcript_extraction"] = "success (yt-dlp_loader)"
                        transcript = transcript_content
                    else:
                        log_and_print("❌ Tier 1 failed: No valid transcript from yt-dlp loader")
                        logs.append("❌ Tier 1 failed: No valid transcript from yt-dlp loader")
                        transcript_content = ""

                except Exception as e:
                    log_and_print(f"❌ Tier 1 failed: {str(e)}")
                    logs.append(f"❌ Tier 1 failed: {str(e)}")
                    transcript_content = ""

                # Tier 2: Fallback to Gemini direct URL processing
                if not transcript_content:
                    if not os.getenv("GEMINI_API_KEY"):
                        error_msg = "yt-dlp loader failed and GEMINI_API_KEY not configured"
                        processing_details["transcript_extraction"] = f"failed: {error_msg}"
                        log_and_print(f"❌ {error_msg}")
                        logs.append(f"❌ {error_msg}")
                    else:
                        log_and_print("🤖 Tier 2: Using Gemini direct URL processing...")
                        logs.append("🤖 Tier 2: Using Gemini direct URL processing...")

                        try:
                            analysis_result = summarize_video(cleaned_url)  # Pass URL directly to Gemini

                            # Extract info from analysis
                            title = analysis_result.title if analysis_result.title else "Unknown Title"
                            author = "Unknown Author"  # Gemini doesn't provide author from URL

                            # Create transcript-like content from analysis
                            transcript_parts = [f"Video Analysis: {analysis_result.title}", "", "Overall Summary:", analysis_result.overall_summary, "", "Key Points:"]

                            for chapter in analysis_result.chapters:
                                transcript_parts.extend([f"\n{chapter.header}:", chapter.summary])

                            transcript = "\n".join(transcript_parts)
                            processing_details["transcript_extraction"] = "success (gemini_direct)"

                            log_and_print("✅ Tier 2 successful: Got content from Gemini direct processing")
                            logs.append("✅ Tier 2 successful: Got content from Gemini direct processing")

                        except Exception as gemini_error:
                            error_msg = f"❌ Tier 2 failed: {str(gemini_error)}"
                            processing_details["transcript_extraction"] = f"failed: {str(gemini_error)}"
                            log_and_print(error_msg)
                            logs.append(error_msg)
                            # Don't fail entire request, continue without transcript
                            transcript = f"[Transcript extraction failed: {str(gemini_error)}]"

                # Update video info with extracted title/author if available
                if video_info and title != "Unknown Title":
                    video_info["title"] = title
                if video_info and author != "Unknown Author":
                    video_info["author"] = author

            except Exception as e:
                processing_details["transcript_extraction"] = f"failed: {str(e)}"
                error_msg = f"Transcript extraction failed: {str(e)}"
                log_and_print(f"❌ {error_msg}")
                logs.append(f"❌ {error_msg}")
                transcript = f"[Transcript extraction failed: {str(e)}]"

        # Step 4: Summary and Analysis Generation (if requested)
        if request.include_summary or request.include_analysis:
            log_and_print("🤖 Step 4: Generating AI summary and analysis...")
            logs.append("🤖 Step 4: Generating AI summary and analysis...")

            try:
                if not os.getenv("GEMINI_API_KEY"):
                    error_msg = "GEMINI_API_KEY not configured"
                    processing_details["summary_generation"] = f"failed: {error_msg}"
                    log_and_print(f"❌ {error_msg}")
                    logs.append(f"❌ {error_msg}")
                    summary = f"[Summary generation failed: {error_msg}]"
                    analysis = {"error": error_msg}
                else:
                    # Determine what content to analyze
                    content_for_analysis = transcript if transcript and not transcript.startswith("[") else cleaned_url

                    if transcript and not transcript.startswith("["):
                        log_and_print("📋 Using extracted transcript for analysis...")
                        logs.append("📋 Using extracted transcript for analysis...")
                        full_content = f"Title: {video_info.get('title', 'Unknown') if video_info else 'Unknown'}\nTranscript:\n{transcript}"
                        analysis_result = summarize_video(full_content)
                    else:
                        log_and_print("🔗 Using direct URL for analysis...")
                        logs.append("🔗 Using direct URL for analysis...")
                        analysis_result = summarize_video(cleaned_url)

                    # Generate formatted summary
                    if request.include_summary:
                        summary_parts = [f"**{analysis_result.title}**", "", "**Overall Summary:**", analysis_result.overall_summary, "", "**Key Takeaways:**"]
                        summary_parts.extend([f"• {takeaway}" for takeaway in analysis_result.takeaways])

                        if analysis_result.key_facts:
                            summary_parts.extend(["", "**Key Facts:**"])
                            summary_parts.extend([f"• {fact}" for fact in analysis_result.key_facts])

                        if analysis_result.chapters:
                            summary_parts.extend(["", "**Chapters:**"])
                            for chapter in analysis_result.chapters:
                                summary_parts.extend([f"**{chapter.header}**", chapter.summary, ""])

                        summary = "\n".join(summary_parts)

                    # Generate structured analysis
                    if request.include_analysis:
                        analysis = {"title": analysis_result.title, "overall_summary": analysis_result.overall_summary, "chapters": [{"header": c.header, "summary": c.summary, "key_points": c.key_points} for c in analysis_result.chapters], "key_facts": analysis_result.key_facts, "takeaways": analysis_result.takeaways, "chapter_count": len(analysis_result.chapters), "total_key_facts": len(analysis_result.key_facts), "total_takeaways": len(analysis_result.takeaways)}

                    processing_details["summary_generation"] = "success"
                    log_and_print("✅ Summary and analysis generated successfully")
                    logs.append("✅ Summary and analysis generated successfully")

            except Exception as e:
                processing_details["summary_generation"] = f"failed: {str(e)}"
                error_msg = f"Summary generation failed: {str(e)}"
                log_and_print(f"❌ {error_msg}")
                logs.append(error_msg)
                summary = f"[Summary generation failed: {str(e)}]"
                analysis = {"error": str(e)}

        # Final processing details
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    log_and_print(f"🚀 Starting YouTube Summarizer API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
