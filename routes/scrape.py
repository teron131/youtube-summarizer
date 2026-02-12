"""Video scraping endpoint for extracting YouTube metadata and transcripts."""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from youtube_summarizer.scrapper import extract_transcript_text, has_transcript_provider_key
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

from .errors import handle_exception, require_env_key
from .helpers import get_processing_time, run_async_task
from .schema import ScrapResponse, YouTubeRequest

router = APIRouter()


@router.post("/scrape", response_model=ScrapResponse)
async def scrap_video(request: YouTubeRequest):
    if not has_transcript_provider_key():
        require_env_key("SCRAPECREATORS_API_KEY")
    start_time = datetime.now(UTC)

    try:
        url = request.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        if not is_youtube_url(url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        url = clean_youtube_url(url)

        transcript = await run_async_task(extract_transcript_text, url)

        data = {
            "url": url,
            "transcript": transcript,
        }

        return ScrapResponse(
            status="success",
            message="Video scraped successfully",
            url=data.get("url"),
            transcript=data.get("transcript"),
            processing_time=get_processing_time(start_time),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception(e, "Scraping") from e
