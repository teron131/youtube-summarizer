"""Video scraping endpoint for extracting YouTube metadata and transcripts."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from routes.schema import ScrapResponse, YouTubeRequest
from youtube_summarizer.scrapper import scrap_youtube
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

from .errors import handle_exception, require_env_key
from .helpers import get_processing_time, parse_scraper_result, run_async_task

router = APIRouter()


@router.post("/scrap", response_model=ScrapResponse)
async def scrap_video(request: YouTubeRequest):
    require_env_key("SCRAPECREATORS_API_KEY")
    start_time = datetime.now(datetime.UTC)

    try:
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        if not is_youtube_url(url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        url = clean_youtube_url(url)

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
        raise handle_exception(e, "Scraping")
