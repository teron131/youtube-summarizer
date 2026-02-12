"""Supadata transcript provider integration."""

import logging
import os

from dotenv import load_dotenv
import requests

from ..utils import clean_text, clean_youtube_url, is_youtube_url

load_dotenv()

SUPADATA_API_URL = "https://api.supadata.ai/v1/transcript"

logger = logging.getLogger(__name__)


def get_supadata_api_key() -> str | None:
    """Return Supadata API key with backwards-compatible env var support."""
    return os.getenv("SUPADATA_API_KEY") or os.getenv("SUPDADATA_API_KEY")


def fetch_supadata_transcript(youtube_url: str, lang: str = "en") -> str | None:
    """Fetch transcript text from Supadata.

    Args:
        youtube_url: YouTube video URL.
        lang: Language code for transcript request.

    Returns:
        Full transcript text when available, otherwise None.
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    api_key = get_supadata_api_key()
    if not api_key:
        return None

    url = clean_youtube_url(youtube_url)
    params = {
        "url": url,
        "lang": lang,
        # Return plain full transcript text instead of timestamped chunks.
        "text": True,
    }
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(SUPADATA_API_URL, params=params, headers=headers, timeout=60)
        if not response.ok:
            logger.warning("Supadata API error: %s %s", response.status_code, response.reason)
            return None

        data = response.json()

        text_blob = data.get("text")
        if isinstance(text_blob, str) and text_blob.strip():
            return clean_text(text_blob)

        content = data.get("content", [])
        if not isinstance(content, list):
            return None

        full_text = " ".join(item.get("text", "").strip() for item in content if isinstance(item, dict))
        return clean_text(full_text) if full_text.strip() else None
    except requests.RequestException as exc:
        logger.warning("Supadata request failed: %s", exc)
        return None
