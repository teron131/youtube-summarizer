"""Scrape Creators transcript-only provider."""

import httpx
from pydantic import BaseModel, ConfigDict

from ..settings import get_settings
from ..utils import clean_text, clean_youtube_url, is_youtube_url


class Channel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    url: str | None = None
    handle: str | None = None
    title: str | None = None


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    startMs: str | None = None
    endMs: str | None = None
    startTimeText: str | None = None


class YouTubeScrapperResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    url: str | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_only_text: str | None = None

    @property
    def parsed_transcript(self) -> str | None:
        """Return cleaned transcript text or None if unavailable."""
        if not self.transcript_only_text or not self.transcript_only_text.strip():
            return None
        return clean_text(self.transcript_only_text)

    @property
    def has_transcript(self) -> bool:
        """Check if video has a transcript available."""
        return bool(self.parsed_transcript)


async def scrape_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """Fetch transcript from Scrape Creators transcript endpoint.

    Uses the transcript-only endpoint:
    https://api.scrapecreators.com/v1/youtube/video/transcript

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        YouTubeScrapperResult: Parsed transcript response
    """
    settings = get_settings()
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    if not settings.scrapecreators_api_key:
        raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")

    youtube_url = clean_youtube_url(youtube_url)

    url = f"{settings.scrapecreators_transcript_url}?url={youtube_url}"
    headers = {"x-api-key": settings.scrapecreators_api_key}
    async with httpx.AsyncClient(timeout=settings.scrape_timeout_seconds) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()

    result = response.json()
    return YouTubeScrapperResult.model_validate(result)
