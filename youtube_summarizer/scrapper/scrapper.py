"""YouTube transcript provider with API-first strategy and yt-dlp/Whisper fallback."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import requests

from ..transcriber.youtube_loader import youtube_loader as transcriber_youtube_loader
from ..utils import clean_text, clean_youtube_url, extract_video_id, is_youtube_url

load_dotenv()

SCRAPECREATORS_ENDPOINT = "https://api.scrapecreators.com/v1/youtube/video/transcript"
SUPADATA_ENDPOINT = "https://api.supadata.ai/v1/transcript"
DEFAULT_TIMEOUT_S = 30


class Channel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    url: str | None = None
    handle: str | None = None
    title: str | None = None


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    startMs: float | None = None
    endMs: float | None = None
    startTimeText: str | None = None


class YouTubeScrapperResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    success: bool | None = None
    credits_remaining: float | None = None
    type: str | None = None
    transcript: list[TranscriptSegment] | None = None
    transcript_only_text: str | None = None
    title: str | None = None
    description: str | None = None
    thumbnail: str | None = None
    url: str | None = None
    id: str | None = None
    viewCountInt: int | None = None
    likeCountInt: int | None = None
    publishDate: str | None = None
    publishDateText: str | None = None
    channel: Channel | None = None
    durationFormatted: str | None = None
    keywords: list[str] | None = None
    videoId: str | None = None
    captionTracks: list[dict[str, Any]] | None = None
    language: str | None = None
    availableLangs: list[str] | None = None

    @property
    def parsed_transcript(self) -> str | None:
        if self.transcript:
            return clean_text(" ".join(seg.text for seg in self.transcript if seg.text))
        if self.transcript_only_text and self.transcript_only_text.strip():
            return clean_text(self.transcript_only_text)
        return None

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript or (self.transcript_only_text and self.transcript_only_text.strip()))


def _get_api_key(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip()
    return value or None


def has_transcript_provider_key() -> bool:
    return bool(_get_api_key("SCRAPECREATORS_API_KEY") or _get_api_key("SUPADATA_API_KEY"))


def _fetch_scrape_creators(video_url: str) -> YouTubeScrapperResult | None:
    api_key = _get_api_key("SCRAPECREATORS_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.get(
            SCRAPECREATORS_ENDPOINT,
            headers={"x-api-key": api_key},
            params={"url": video_url},
            timeout=DEFAULT_TIMEOUT_S,
        )
    except requests.RequestException:
        return None

    if response.status_code in {401, 403} or not response.ok:
        return None

    try:
        data: dict[str, Any] = response.json()
        return YouTubeScrapperResult.model_validate(data)
    except Exception:
        return None


def _fetch_supadata(video_url: str) -> YouTubeScrapperResult | None:
    api_key = _get_api_key("SUPADATA_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.get(
            SUPADATA_ENDPOINT,
            headers={"x-api-key": api_key},
            params={"url": video_url, "lang": "en", "text": "true", "mode": "auto"},
            timeout=DEFAULT_TIMEOUT_S,
        )
    except requests.RequestException:
        return None

    if response.status_code in {401, 403} or response.status_code == 202 or not response.ok:
        return None

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        return None

    content = data.get("content")
    transcript_only_text: str | None = content if isinstance(content, str) else None
    transcript: list[TranscriptSegment] | None = None

    if transcript_only_text is None and isinstance(content, list):
        transcript = []
        for item in content:
            if not isinstance(item, dict):
                continue
            transcript.append(
                TranscriptSegment(
                    text=item.get("text"),
                    startMs=item.get("offset"),
                    endMs=(item.get("offset", 0) or 0) + (item.get("duration", 0) or 0),
                    startTimeText=None,
                )
            )

    return YouTubeScrapperResult(
        url=video_url,
        transcript=transcript,
        transcript_only_text=transcript_only_text,
        videoId=extract_video_id(video_url),
        language=data.get("lang"),
        availableLangs=data.get("availableLangs"),
        success=True,
        type="video",
    )


def scrape_youtube(youtube_url: str) -> YouTubeScrapperResult:
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)

    result = _fetch_scrape_creators(youtube_url)
    if result and result.has_transcript:
        return result

    result = _fetch_supadata(youtube_url)
    if result and result.has_transcript:
        return result

    if not has_transcript_provider_key():
        raise ValueError("No API keys found for Scrape Creators or Supadata")

    raise ValueError("Failed to fetch transcript from available providers")


def _extract_fallback_subtitle(content: str) -> str | None:
    marker = "Subtitle:\n"
    if marker not in content:
        return None
    subtitle = content.split(marker, 1)[1].strip()
    return clean_text(subtitle) if subtitle else None


def get_transcript(youtube_url: str) -> str:
    result = scrape_youtube(youtube_url)
    if not result.has_transcript:
        raise ValueError("Video has no transcript")

    transcript = result.parsed_transcript
    if not transcript:
        raise ValueError("Transcript is empty")
    return transcript


def extract_transcript_text(youtube_url: str) -> str:
    try:
        return get_transcript(youtube_url)
    except Exception:
        fallback_content = transcriber_youtube_loader(youtube_url)
        subtitle = _extract_fallback_subtitle(fallback_content)
        if subtitle:
            return subtitle
        raise
