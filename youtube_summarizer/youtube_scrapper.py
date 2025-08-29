"""
This module uses the Apify YouTube scraper API to scrape a YouTube video and return the transcript and other metadata.
https://apify.com/scrape-creators/best-youtube-scraper
The API result is wrapped by YouTubeScrapperResult object.

Important video metadata:
result.title: str = 'The Trillion Dollar Equation'
result.thumbnail: str = 'https://img.youtube.com/vi/A5w-dEgIU1M/maxresdefault.jpg'
result.channel.title: str = 'NVIDIA Developer'
result.durationFormatted: str = '00:06:32'
result.viewCountInt: int = 13462116
result.likeCountInt: int = 321234
result.publishDateText: str = 'Feb 27, 2024'
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

    @property
    def parsed_transcript(self) -> str:
        """Parse transcript into a single string with chapter formatting.

        If chapters are present, the transcript is segmented by chapter, formatted as
        '## <Chapter Title>' followed by the cleaned-up transcript text for that chapter.
        If no chapters are available, the entire transcript is cleaned and returned as a single block.

        Returns:
            A formatted string with the transcript, organized by chapters if available.
        """
        if not self.chapters:
            return clean_text(self.transcript_only_text)

        # Create chapter windows and a list of their start times for binary search.
        windows = [ChapterTranscript(title=chapter.title, start_ms=chapter.startSeconds * 1000) for chapter in self.chapters]
        chapter_start_times = [window.start_ms for window in windows]

        # Assign each transcript segment to its corresponding chapter window using binary search.
        # This is efficient as it avoids nested loops for chapter lookups.
        for seg in self.transcript:
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

    @property
    def timestamped_transcript(self) -> str:
        """Parse transcript into individual sentences with preserved starting timestamps.

        Splits transcript by periods to create individual sentences, each with
        its own timestamp based on when that sentence begins in the video.
        Sentences are formatted as '[MM:SS] Individual sentence.'

        Returns:
            A formatted string with individual sentences and their corresponding timestamps.
        """
        if not self.transcript:
            return ""

        # Build segments with timestamps using list comprehension
        valid_segments = [(seg.text.strip(), int(seg.startMs)) for seg in self.transcript if seg.text and seg.text.strip()]

        if not valid_segments:
            return ""

        # Build character-to-timestamp mapping using comprehensions
        text_parts = [" " + seg_text for seg_text, timestamp in valid_segments]

        # Create position offsets for each segment using cumulative sum approach
        segment_positions = [sum(len(text_parts[i]) for i in range(j)) for j in range(len(text_parts) + 1)]

        # Build character-to-timestamp mapping using dict comprehension
        char_to_timestamp = {segment_positions[idx] + i: timestamp for idx, (seg_text, timestamp) in enumerate(valid_segments) for i in range(len(" " + seg_text))}

        # Clean the full text and split by periods using list comprehension
        full_text = clean_text("".join(text_parts))
        sentence_parts = [s.strip() for s in full_text.split(".") if s.strip()]

        def _format_timestamp(timestamp_ms: int) -> str:
            seconds = timestamp_ms // 1000
            return f"[{seconds // 60:02d}:{seconds % 60:02d}]"

        def _get_sentence_with_timestamp(sentence_data):
            idx, sentence_text = sentence_data
            current_pos = sum(len(s) + 1 for s in sentence_parts[:idx])  # Calculate position

            # Find first available timestamp for this sentence using next()
            sentence_timestamp = next((char_to_timestamp[i] for i in range(current_pos, min(current_pos + len(sentence_text), len(char_to_timestamp))) if i in char_to_timestamp), None)

            return f"{_format_timestamp(sentence_timestamp)} {sentence_text}." if sentence_timestamp else None

        # Generate all sentences using list comprehension and filter out None values
        sentences = [result for result in map(_get_sentence_with_timestamp, enumerate(sentence_parts)) if result]

        return "\n".join(sentences)


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
