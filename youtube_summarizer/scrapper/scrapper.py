"""YouTube transcript provider with API-first strategy and yt-dlp/Whisper fallback."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from ..settings import get_settings
from ..utils import clean_text, clean_youtube_url, extract_video_id, is_youtube_url


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


def has_transcript_provider_key() -> bool:
    settings = get_settings()
    return settings.has_any_transcript_provider


async def _fetch_scrape_creators(
    client: httpx.AsyncClient,
    video_url: str,
) -> YouTubeScrapperResult | None:
    settings = get_settings()
    api_key = settings.scrapecreators_api_key
    if not api_key:
        return None

    try:
        response = await client.get(
            settings.scrapecreators_transcript_url,
            headers={"x-api-key": api_key},
            params={"url": video_url},
        )
    except httpx.HTTPError:
        return None

    if response.status_code in {401, 403} or not response.is_success:
        return None

    try:
        data: dict[str, Any] = response.json()
        return YouTubeScrapperResult.model_validate(data)
    except Exception:
        return None


async def _fetch_supadata(
    client: httpx.AsyncClient,
    video_url: str,
) -> YouTubeScrapperResult | None:
    settings = get_settings()
    api_key = settings.supadata_api_key
    if not api_key:
        return None

    try:
        response = await client.get(
            settings.supadata_transcript_url,
            headers={"x-api-key": api_key},
            params={"url": video_url, "lang": "en", "text": "true", "mode": "auto"},
        )
    except httpx.HTTPError:
        return None

    if response.status_code in {401, 403} or response.status_code == 202 or not response.is_success:
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


async def scrape_youtube(youtube_url: str) -> YouTubeScrapperResult:
    settings = get_settings()
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)

    async with httpx.AsyncClient(timeout=settings.scrape_timeout_seconds) as client:
        result = await _fetch_scrape_creators(client, youtube_url)
        if result and result.has_transcript:
            return result

        result = await _fetch_supadata(client, youtube_url)
        if result and result.has_transcript:
            return result

    if not has_transcript_provider_key():
        raise ValueError("No API keys found for Scrape Creators or Supadata")

    raise ValueError("Failed to fetch transcript from available providers")


async def get_transcript(youtube_url: str) -> str:
    result = await scrape_youtube(youtube_url)
    if not result.has_transcript:
        raise ValueError("Video has no transcript")

    transcript = result.parsed_transcript
    if not transcript:
        raise ValueError("Transcript is empty")
    return transcript


async def extract_transcript_text(youtube_url: str) -> str:
    return await get_transcript(youtube_url)
