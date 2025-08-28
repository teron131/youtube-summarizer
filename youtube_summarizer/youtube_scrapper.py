import json
import os
from datetime import datetime
from typing import List, Optional

import requests
from pydantic import BaseModel

from .utils import clean_text, clean_youtube_url, is_youtube_url

APIFY_API_KEY = os.getenv("APIFY_API_KEY")


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


def youtube_scrap(youtube_url: str) -> YouTubeScrapperResult:
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
    """Parse transcript into a single string

    If chapters are present, format into '## <Chapter>' followed by its transcript text...

    If chapters are not present, clean the the transcript_only_text.

    Args:
        result: YouTubeScrapperResult

    Returns:
        str: A single string with the transcript formatted as described above.
    """
    if not result.chapters:
        # Clean the transcript_only_text using the same regex patterns
        return clean_text(result.transcript_only_text)

    # Build chapter windows with list comprehension and direct attribute access
    num_chapters = len(result.chapters)
    windows = [
        {
            "title": ch.title,
            "start_ms": ch.startSeconds * 1000,
            "end_ms": (result.chapters[i + 1].startSeconds * 1000 if i + 1 < num_chapters else 10**9 * 1000),
            "parts": [],
        }
        for i, ch in enumerate(result.chapters)
    ]

    # Single-pass assignment with direct attribute access
    current_idx = 0
    current_window = windows[0]

    for seg in result.transcript:
        seg_ms = int(seg.startMs)

        # Move to next window if needed
        while current_idx + 1 < len(windows) and seg_ms >= current_window["end_ms"]:
            current_idx += 1
            current_window = windows[current_idx]

        # Add segment text to current window
        if current_window["start_ms"] <= seg_ms < current_window["end_ms"]:
            if text := seg.text.strip():
                current_window["parts"].append(text)

    # Build final output with list comprehension and clean_text function
    result_parts = [
        (lambda title_line, parts: f"{title_line}\n{clean_text(' '.join(parts))}" if parts else title_line)(
            f"## {window['title']}",
            window["parts"],
        )
        for window in windows
    ]

    return "\n\n".join(result_parts)
