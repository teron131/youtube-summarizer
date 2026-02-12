"""Video scraping endpoint for extracting YouTube metadata and transcripts."""

from datetime import UTC, datetime
import logging

from fastapi import APIRouter, HTTPException

from youtube_summarizer.scrapper import scrap_youtube
from youtube_summarizer.transcript_provider import extract_transcript_text, has_transcript_provider_key
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

from .errors import handle_exception, require_env_key
from .helpers import get_processing_time, parse_scraper_result, run_async_task
from .schema import ScrapResponse, YouTubeRequest

router = APIRouter()
logger = logging.getLogger(__name__)


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

        # Preserve metadata when Scrape Creators is configured.
        data = {
            "url": url,
            "title": None,
            "author": None,
            "transcript": transcript,
            "duration": None,
            "thumbnail": None,
            "view_count": None,
            "like_count": None,
            "upload_date": None,
        }
        try:
            result = await run_async_task(scrap_youtube, url)
            metadata_data = parse_scraper_result(result)
            metadata_data["transcript"] = transcript
            data = metadata_data
        except Exception as exc:
            # Supadata-only mode can still provide transcript text without metadata.
            logger.info("Metadata enrichment via Scrape Creators skipped: %s", exc)

        return ScrapResponse(
            status="success",
            message="Video scraped successfully",
            url=data.get("url"),
            title=data.get("title"),
            author=data.get("author"),
            transcript=data.get("transcript"),
            duration=data.get("duration"),
            thumbnail=data.get("thumbnail"),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            upload_date=data.get("upload_date"),
            processing_time=get_processing_time(start_time),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception(e, "Scraping") from e
