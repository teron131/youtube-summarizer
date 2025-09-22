"""
This module uses the Scrape Creators YouTube Video API to scrape a YouTube video and return the transcript and other metadata.
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
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from .utils import clean_text, clean_youtube_url, is_youtube_url

SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")


class Channel(BaseModel):
    id: str
    url: str
    handle: str
    title: str


class WatchNextVideo(BaseModel):
    id: str
    title: Optional[str] = None
    thumbnail: Optional[str] = None
    channel: Optional[Channel] = None
    publishDateText: Optional[str] = None
    publishDate: Optional[str] = None  # API returns ISO string, not datetime
    viewCountText: Optional[str] = None
    viewCountInt: Optional[int] = None
    lengthText: Optional[str] = None
    videoUrl: Optional[str] = None


class TranscriptSegment(BaseModel):
    text: str
    startMs: str
    endMs: str
    startTimeText: str


class YouTubeScrapperResult(BaseModel):
    id: str
    thumbnail: Optional[str] = None
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    commentCountText: Optional[str] = None
    commentCountInt: Optional[int] = None
    likeCountText: Optional[str] = None
    likeCountInt: Optional[int] = None
    viewCountText: Optional[str] = None
    viewCountInt: Optional[int] = None
    publishDateText: Optional[str] = None
    publishDate: Optional[str] = None  # API returns ISO string, not datetime
    channel: Optional[Channel] = None
    watchNextVideos: List[WatchNextVideo] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    durationMs: Optional[int] = None
    durationFormatted: Optional[str] = None
    transcript: Optional[List[TranscriptSegment]] = None
    transcript_only_text: Optional[str] = None

    @property
    def parsed_transcript(self) -> Optional[str]:
        """Parse transcript into a single string.

        Since the API doesn't provide chapter information, this simply returns
        the cleaned transcript_only_text if available.

        Returns:
            A cleaned string with the transcript text.
            Returns None if no transcript is available.
        """
        # Handle case where transcript_only_text is None or empty
        if self.transcript_only_text is None or not self.transcript_only_text.strip():
            return None
        return clean_text(self.transcript_only_text)

    @property
    def has_transcript(self) -> bool:
        """Check if this video has a transcript available."""
        return self.transcript is not None and len(self.transcript) > 0 and self.transcript_only_text is not None and self.transcript_only_text.strip() != ""


def scrap_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """
    Scrape a YouTube video and return the transcript and other metadata.

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
    headers = {"x-api-key": os.getenv("SCRAPECREATORS_API_KEY")}
    response = requests.get(url, headers=headers)

    result = response.json()
    return YouTubeScrapperResult.model_validate(result)
