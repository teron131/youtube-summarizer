"""Video scraping route handlers"""

from datetime import datetime
from fastapi import HTTPException
from routes.schema import ScrapResponse, YouTubeRequest

from youtube_summarizer.scrapper import scrap_youtube
from .errors import handle_exception, require_env_key
from .helpers import get_processing_time, parse_scraper_result, run_async_task
from .validation import validate_url


async def scrap_video_handler(request: YouTubeRequest) -> ScrapResponse:
    """Extract video metadata and transcript."""
    require_env_key("SCRAPECREATORS_API_KEY")

    start_time = datetime.now()

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
        raise handle_exception(e, "Scraping")
