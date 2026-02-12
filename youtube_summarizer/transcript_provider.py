"""Transcript provider orchestration for YouTube URLs."""

import logging
import os

from .scrapper import YouTubeScrapperResult, scrap_youtube
from .supadata import fetch_supadata_transcript, get_supadata_api_key
from .utils import is_youtube_url

logger = logging.getLogger(__name__)

SCRAPECREATORS_PROVIDER = "scrapecreators"
SUPADATA_PROVIDER = "supadata"


def _provider_order() -> tuple[str, str]:
    preference = os.getenv("TRANSCRIPT_PROVIDER_PREFERENCE", SCRAPECREATORS_PROVIDER).strip().lower()
    if preference == SUPADATA_PROVIDER:
        return (SUPADATA_PROVIDER, SCRAPECREATORS_PROVIDER)
    return (SCRAPECREATORS_PROVIDER, SUPADATA_PROVIDER)


def has_transcript_provider_key() -> bool:
    """Return True when at least one transcript provider key is available."""
    return bool(os.getenv("SCRAPECREATORS_API_KEY") or get_supadata_api_key())


def _from_scrapecreators(youtube_url: str) -> str | None:
    result: YouTubeScrapperResult = scrap_youtube(youtube_url)
    if not result.transcript_only_text or not result.transcript_only_text.strip():
        return None
    return result.parsed_transcript


def _from_supadata(youtube_url: str) -> str | None:
    return fetch_supadata_transcript(youtube_url)


def extract_transcript_text(youtube_url: str) -> str:
    """Extract transcript full text using configured provider order.

    Returns full transcript text only (no timestamps).
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    attempts: list[str] = []
    for provider in _provider_order():
        try:
            transcript = _from_scrapecreators(youtube_url) if provider == SCRAPECREATORS_PROVIDER else _from_supadata(youtube_url)
            if transcript and transcript.strip():
                logger.info("Transcript resolved via %s", provider)
                return transcript
            attempts.append(f"{provider}:empty")
        except Exception as exc:
            logger.warning("Transcript provider %s failed: %s", provider, exc)
            attempts.append(f"{provider}:error")

    raise ValueError(f"Video has no transcript. Attempts: {', '.join(attempts)}")
