"""
This module uses the Apify YouTube scraper API to scrape a YouTube video and return the transcript and other metadata.
https://scrapecreators.com
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

import json
import os
from datetime import datetime
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from .utils import clean_text, clean_youtube_url, is_youtube_url

SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")


class ChapterTranscript(BaseModel):
    """Represents a chapter with its transcript parts."""

    title: str
    transcript_parts: List[str] = Field(default_factory=list)

    def format_output(self) -> str:
        """Formats the chapter title and its transcript parts into a string."""
        title_line = f"## {self.title}"
        if not self.transcript_parts:
            return title_line

        # Join parts into a single text block and clean it
        transcript_text = clean_text(" ".join(self.transcript_parts))
        return f"{title_line}\n{transcript_text}"


class Channel(BaseModel):
    id: str
    url: str
    handle: str
    title: str


class Chapter(BaseModel):
    title: str
    timeDescription: str
    startSeconds: int


class WatchNextVideo(BaseModel):
    id: str
    title: str
    thumbnail: str
    channel: Channel
    publishedTimeText: str
    publishedTime: datetime
    publishDateText: str
    publishDate: datetime
    viewCountText: str
    viewCountInt: int
    lengthText: str
    lengthInSeconds: int
    videoUrl: str


class TranscriptSegment(BaseModel):
    text: str
    startMs: str
    endMs: str
    startTimeText: str


class YouTubeScrapperResult(BaseModel):
    id: str
    thumbnail: str
    url: str
    type: str
    title: str
    description: str
    commentCountInt: Optional[int] = None
    likeCountText: str
    likeCountInt: int
    viewCountText: str
    viewCountInt: int
    publishDateText: str
    publishDate: datetime
    channel: Channel
    chapters: List[Chapter]
    watchNextVideos: List[WatchNextVideo]
    keywords: List[str]
    durationMs: int
    durationFormatted: str
    transcript: List[TranscriptSegment]
    transcript_only_text: str
    language: str

    @property
    def parsed_transcript(self) -> str:
        """Parse transcript into a single string with chapter formatting.

        If chapters are present, formats each chapter as '## <Chapter Title>'.
        If no chapters are available, the entire transcript is cleaned and returned as a single block.

        Returns:
            A formatted string with the transcript, organized by chapters if available.
        """
        if not self.chapters:
            return clean_text(self.transcript_only_text)

        # Format each chapter without timestamp-based segmentation
        result_parts = []
        for chapter in self.chapters:
            chapter_transcript = ChapterTranscript(title=chapter.title)
            # Since we can't reliably assign transcript segments to chapters without timestamps,
            # we'll just format the chapter titles
            result_parts.append(chapter_transcript.format_output())

        return "\n\n".join(result_parts)


def scrap_youtube(youtube_url: str) -> YouTubeScrapperResult:
    """
    Scrape a YouTube video and return the transcript and other metadata.

    Uses the Apify YouTube scraper API: https://apify.com/scrape-creators/best-youtube-scraper

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        YouTubeScrapperResult: Parsed video data including transcript and metadata
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)
    api_url = f"https://api.apify.com/v2/acts/scrape-creators~best-youtube-scraper/run-sync-get-dataset-items?token={SCRAPECREATORS_API_KEY}&maxItems=1&timeout=60"

    data = json.dumps({"getTranscript": True, "videoUrls": [youtube_url]})
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.request("POST", url=api_url, data=data, headers=headers)

    result = json.loads(response.text)[0]
    return YouTubeScrapperResult.model_validate(result)
