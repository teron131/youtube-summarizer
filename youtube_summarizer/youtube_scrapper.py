import json
import os
from datetime import datetime
from typing import List, Optional

import requests
from pydantic import BaseModel

from .utils import clean_youtube_url, is_youtube_url

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
