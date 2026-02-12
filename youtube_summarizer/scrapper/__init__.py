"""Scrapper package exports."""

from .scrapper import (
    Channel,
    TranscriptSegment,
    YouTubeScrapperResult,
    extract_transcript_text,
    has_transcript_provider_key,
    scrap_youtube,
    scrape_youtube,
)

__all__ = [
    "Channel",
    "TranscriptSegment",
    "YouTubeScrapperResult",
    "extract_transcript_text",
    "has_transcript_provider_key",
    "scrap_youtube",
    "scrape_youtube",
]
