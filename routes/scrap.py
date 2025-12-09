"""
Video scraping route handlers
"""

import os
from datetime import datetime

from fastapi import HTTPException
from models.requests import ScrapResponse, YouTubeRequest
from youtube_summarizer.utils import log_and_print
from youtube_summarizer.youtube_scrapper import scrap_youtube

from .helpers import get_processing_time, parse_scraper_result, run_async_task
from .validation import validate_url


async def scrap_video_handler(request: YouTubeRequest) -> ScrapResponse:
    """Extract video metadata and transcript."""
    start_time = datetime.now()

    if not os.getenv("SCRAPECREATORS_API_KEY"):
        raise HTTPException(status_code=500, detail="Required API key missing")

    try:
        url = validate_url(request.url)
        result = await run_async_task(scrap_youtube, url)
        data = parse_scraper_result(result)

        return ScrapResponse(
            status="success",
            message="Video scraped successfully",
            url=data["url"],
            title=data["title"],
            author=data["author"],
            transcript=data["transcript"],
            duration=data["duration"],
            thumbnail=data["thumbnail"],
            view_count=data["view_count"],
            like_count=data["like_count"],
            upload_date=data["upload_date"],
            processing_time=get_processing_time(start_time),
        )
    except HTTPException:
        raise
    except Exception as e:
        log_and_print(f"❌ Scraping failed: {str(e)}")
        if "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail="API quota exceeded")
        elif "400" in str(e) or "Invalid" in str(e):
            raise HTTPException(status_code=400, detail=f"Invalid URL: {str(e)[:100]}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:100]}")
