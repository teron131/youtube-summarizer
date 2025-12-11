"""This module uses the Scrape Creators YouTube Video API to scrape a YouTube video and return the transcript and other metadata.

Docs: https://docs.scrapecreators.com/v1/youtube/video
The API result is wrapped by YouTubeScrapperResult object.

Important video metadata:
result.title: str = 'The Trillion Dollar Equation'
result.thumbnail: str = 'https://img.youtube.com/vi/A5w-dEgIU1M/maxresdefault.jpg'
result.channel.title: str = 'NVIDIA Developer'
result.durationFormatted: str = '00:06:32'
result.publishDateText: str = 'Feb 27, 2024'
result.viewCountInt: int = 13462116
result.likeCountInt: int = 321234
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import requests

from .utils import clean_text, clean_youtube_url, is_youtube_url

load_dotenv()

SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")


class Channel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    startMs: str | None = None
    endMs: str | None = None
    startTimeText: str | None = None


class YouTubeScrapperResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = None
    thumbnail: str | None = None
    title: str | None = None
    description: str | None = None
    likeCountInt: int | None = None
    viewCountInt: int | None = None
    publishDateText: str | None = None
    channel: Channel | None = None
    durationFormatted: str | None = None
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
        return bool(self.transcript and self.transcript_only_text and self.transcript_only_text.strip())


def scrap_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """Scrape a YouTube video and return the transcript and other metadata.

    Uses the Scrape Creators YouTube API: https://api.scrapecreators.com/v1/youtube/video

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        YouTubeScrapperResult: Parsed video data including transcript and metadata
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    if not SCRAPECREATORS_API_KEY:
        raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")

    youtube_url = clean_youtube_url(youtube_url)

    url = f"https://api.scrapecreators.com/v1/youtube/video?url={youtube_url}&get_transcript=true"
    headers = {"x-api-key": SCRAPECREATORS_API_KEY}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    result = response.json()
    return YouTubeScrapperResult.model_validate(result)
