"""
This module contains helper functions for string manipulation, data parsing, and logging, used across the YouTube Summarizer application.
"""

import logging
import re
from typing import Any

from opencc import OpenCC

_OPENCC_S2HK = OpenCC("s2hk")
YOUTUBE_URL_PATTERN = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+")
YOUTUBE_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/)([\w-]+)")


def serialize_nested(obj: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """Serialize nested objects with depth limit to avoid recursion errors."""
    if depth > max_depth:
        return str(obj)[:100] + "...[deep nesting]"

    if isinstance(obj, dict):
        return {k: serialize_nested(v, depth + 1, max_depth) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_nested(item, depth + 1, max_depth) for item in obj]
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    else:
        return obj


def log_and_print(message: str):
    """Log message. (Deprecated: use logging.info directly)"""
    logging.info(message)


# Module-level compiled patterns for maximum performance
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and newlines."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube watch or short link."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


def _extract_video_id(url: str) -> str | None:
    match = YOUTUBE_ID_PATTERN.search(url)
    return match.group(1) if match else None


def clean_youtube_url(url: str) -> str:
    """Normalize YouTube URLs to https://www.youtube.com/watch?v=<id> when possible."""
    video_id = _extract_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else url


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Hong Kong variant)."""
    return _OPENCC_S2HK.convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    chunks = result.get("chunks") or []
    lines = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text")]
    return s2hk("\n".join(lines))
