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
    # Try attribute access first
    if hasattr(result, "parsed_transcript") and result.parsed_transcript and result.parsed_transcript.strip():
        return result.parsed_transcript

    # Try dictionary/model dump
    data = result.model_dump() if hasattr(result, "model_dump") else (result if isinstance(result, dict) else {})
    transcript = data.get("parsed_transcript") or data.get("transcript_only_text")

    return transcript if transcript and transcript.strip() else None


def parse_scraper_result(result) -> dict[str, Any]:
    try:
        data = result.model_dump() if hasattr(result, "model_dump") else (result if isinstance(result, dict) else {})
        channel = data.get("channel", {})

        url = data.get("url")
        if not url and data.get("id"):
            url = f"https://www.youtube.com/watch?v={data.get('id')}"

        return {
            "url": url,
            "title": data.get("title"),
            "author": channel.get("title") if isinstance(channel, dict) else None,
            "transcript": _get_transcript(result),
            "duration": data.get("durationFormatted"),
            "thumbnail": data.get("thumbnail"),
            "view_count": data.get("viewCountInt"),
            "like_count": data.get("likeCountInt"),
            "upload_date": data.get("publishDateText"),
        }
    except Exception as e:
        logging.warning(f"Error parsing scraper result: {str(e)}")
        return {"url": None, "title": None, "author": None, "transcript": _get_transcript(result), "duration": None, "thumbnail": None, "view_count": None, "like_count": None, "upload_date": None}
