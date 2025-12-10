"""Helper functions for API routes"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import HTTPException

TIMEOUT_LONG = 300.0


async def run_async_task(func, *args, timeout: float = TIMEOUT_LONG):
    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, func, *args),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {timeout} seconds",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:100]}")


def get_processing_time(start_time: datetime) -> str:
    return f"{(datetime.now() - start_time).total_seconds():.1f}s"


def _get_transcript(result) -> str | None:
    if hasattr(result, "parsed_transcript"):
        transcript = result.parsed_transcript
        if transcript and transcript.strip():
            return transcript

    if hasattr(result, "model_dump"):
        result_dict = result.model_dump()
        transcript = result_dict.get("parsed_transcript") or result_dict.get(
            "transcript_only_text"
        )
        if transcript and transcript.strip():
            return transcript

    return None


def parse_scraper_result(result) -> dict[str, Any]:
    default = {
        "url": None,
        "title": None,
        "author": None,
        "transcript": None,
        "duration": None,
        "thumbnail": None,
        "view_count": None,
        "like_count": None,
        "upload_date": None,
    }

    try:
        result_dict = result.model_dump() if hasattr(result, "model_dump") else {}
        channel = result_dict.get("channel", {})

        url = result_dict.get("url")
        if not url and result_dict.get("id"):
            url = f"https://www.youtube.com/watch?v={result_dict.get('id')}"

        return {
            "url": url,
            "title": result_dict.get("title"),
            "author": channel.get("title") if isinstance(channel, dict) else None,
            "transcript": _get_transcript(result),
            "duration": result_dict.get("durationFormatted"),
            "thumbnail": result_dict.get("thumbnail"),
            "view_count": result_dict.get("viewCountInt"),
            "like_count": result_dict.get("likeCountInt"),
            "upload_date": result_dict.get("publishDateText"),
        }
    except Exception as e:
        logging.warning(f"Error parsing scraper result: {str(e)}")
        default["transcript"] = _get_transcript(result)
        return default
