"""Helper functions for string manipulation, data parsing, and Chinese text conversion."""

import re
from typing import Any

from opencc import OpenCC

# Regex patterns
YOUTUBE_URL_PATTERN = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+")
YOUTUBE_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/)([\w-]+)")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Chinese converter
_OPENCC_S2HK = OpenCC("s2hk")


def serialize_nested(obj: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """Serialize nested objects with depth limit to avoid recursion errors."""
    if depth > max_depth:
        return str(obj)[:100] + "...[deep nesting]"

    if isinstance(obj, dict):
        return {k: serialize_nested(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_nested(item, depth + 1, max_depth) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and newlines."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube watch or short link."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


def clean_youtube_url(url: str) -> str:
    """Normalize YouTube URLs to https://www.youtube.com/watch?v=<id> when possible."""
    match = YOUTUBE_ID_PATTERN.search(url)
    if not match:
        return url

    video_id = match.group(1)
    return f"https://www.youtube.com/watch?v={video_id}"


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Hong Kong variant)."""
    return _OPENCC_S2HK.convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    chunks = result.get("chunks") or []
    lines = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text")]
    return s2hk("\n".join(lines))
