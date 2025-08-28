"""
This module uses the Apify YouTube scraper API to scrape a YouTube video and return the transcript and other metadata.
https://apify.com/scrape-creators/best-youtube-scraper
The API result is wrapped by YouTubeScrapperResult object.

Important video metadata:
result.title: str = 'Getting Started with the NVIDIA Jetson AGX Thor Developer Kit for Physical AI'
result.thumbnail: str = 'https://img.youtube.com/vi/iYT2haVIgSM/maxresdefault.jpg'
result.channel.title: str = 'NVIDIA Developer'
result.durationFormatted: str = '00:06:32'
result.likeCountInt: int = 590
result.publishDateText: str = 'Aug 25, 2025'
"""

import bisect
import json
import os
from datetime import datetime
from typing import List, Optional

import requests
from pydantic import BaseModel, Field

from .utils import clean_text, clean_youtube_url, is_youtube_url

APIFY_API_KEY = os.getenv("APIFY_API_KEY")


class ChapterTranscript(BaseModel):
    """Represents a time window for one video chapter."""

    title: str
    start_ms: int
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
    api_url = f"https://api.apify.com/v2/acts/scrape-creators~best-youtube-scraper/run-sync-get-dataset-items?token={APIFY_API_KEY}&maxItems=1&timeout=60"

    data = json.dumps({"getTranscript": True, "videoUrls": [youtube_url]})
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.request("POST", url=api_url, data=data, headers=headers)

    result = json.loads(response.text)[0]
    return YouTubeScrapperResult.model_validate(result)


def parse_transcript(result: YouTubeScrapperResult) -> str:
    """Parse transcript into a single string with chapter formatting.

    If chapters are present, the transcript is segmented by chapter, formatted as
    '## <Chapter Title>' followed by the cleaned-up transcript text for that chapter.
    If no chapters are available, the entire transcript is cleaned and returned as a single block.

    Args:
        result: The YouTubeScrapperResult containing transcript and chapter data.

    Returns:
        A formatted string with the transcript, organized by chapters if available.
    """
    if not result.chapters:
        return clean_text(result.transcript_only_text)

    # Create chapter windows and a list of their start times for binary search.
    windows = [ChapterTranscript(title=chapter.title, start_ms=chapter.startSeconds * 1000) for chapter in result.chapters]
    chapter_start_times = [window.start_ms for window in windows]

    # Assign each transcript segment to its corresponding chapter window using binary search.
    # This is efficient as it avoids nested loops for chapter lookups.
    for seg in result.transcript:
        if not seg.text or not seg.text.strip():
            continue

        seg_ms = int(seg.startMs)

        # Find the index of the chapter this segment belongs to.
        # bisect_right gives the insertion point, so we subtract 1 to get the current chapter.
        idx = bisect.bisect_right(chapter_start_times, seg_ms) - 1

        if idx >= 0:
            windows[idx].transcript_parts.append(seg.text)

    # Format each chapter window into its final string representation.
    result_parts = [window.format_output() for window in windows]

    return "\n\n".join(result_parts)
