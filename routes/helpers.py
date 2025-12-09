"""
Helper functions for API routes
"""

import asyncio
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from youtube_summarizer.utils import log_and_print

TIMEOUT_LONG = 300.0  # 5 minutes for AI processing


async def run_async_task(func, *args, timeout: float = TIMEOUT_LONG):
    """Execute blocking function asynchronously with timeout."""
    try:
        return await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, func, *args), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Request timed out after {timeout} seconds")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:100]}")


def get_processing_time(start_time: datetime) -> str:
    """Calculate and format processing time."""
    return f"{(datetime.now() - start_time).total_seconds():.1f}s"


def parse_scraper_result(result) -> dict[str, Any]:
    """Parse scraper result to dict with error handling."""
    try:
        # Handle both Pydantic objects and raw dict responses
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        else:
            result_dict = result if isinstance(result, dict) else {}

        # Map fields from Scrape Creators API format
        like_count = result_dict.get("likeCountInt")
        upload_date = result_dict.get("publishDateText") or result_dict.get("publishDate")

        # Get transcript with proper fallback chain
        transcript = ""
        if hasattr(result, "parsed_transcript"):
            transcript = result.parsed_transcript
        elif result_dict.get("parsed_transcript"):
            transcript = result_dict["parsed_transcript"]
        elif result_dict.get("transcript_only_text"):
            transcript = result_dict["transcript_only_text"]
        elif hasattr(result, "transcript_only_text"):
            transcript = getattr(result, "transcript_only_text", "")

        # Ensure transcript is not empty
        if not transcript or not transcript.strip():
            transcript = None

        # Extract all fields with fallbacks
        return {
            "url": result_dict.get("url") or None,
            "title": result_dict.get("title") or None,
            "author": result_dict.get("channel", {}).get("title") if isinstance(result_dict.get("channel"), dict) else None,
            "transcript": transcript,
            "duration": result_dict.get("durationFormatted") or None,
            "thumbnail": result_dict.get("thumbnail") or None,
            "view_count": result_dict.get("viewCountInt") or None,
            "like_count": like_count or None,
            "upload_date": upload_date or None,
        }
    except Exception as e:
        log_and_print(f"⚠️ Warning: Error parsing scraper result: {str(e)}")

        # Fallback parsing
        transcript = None
        try:
            if hasattr(result, "parsed_transcript"):
                transcript = result.parsed_transcript
            elif hasattr(result, "transcript_only_text"):
                transcript = getattr(result, "transcript_only_text", "")
            if transcript and not transcript.strip():
                transcript = None
        except:
            transcript = None

        return {
            "url": None,
            "title": None,
            "author": None,
            "transcript": transcript,
            "duration": None,
            "thumbnail": None,
            "view_count": None,
            "like_count": None,
            "upload_date": None,
        }
