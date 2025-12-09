"""
Request validation utilities
"""

from fastapi import HTTPException
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

ERROR_MESSAGES = {
    "invalid_url": "Invalid YouTube URL",
    "empty_url": "URL is required",
    "empty_content": "Content cannot be empty",
}


def validate_url(url: str) -> str:
    """Validate and clean YouTube URL with enhanced checks."""
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["empty_url"])

    url = url.strip()
    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["invalid_url"])

    return clean_youtube_url(url)


def validate_content(content: str) -> str:
    """Validate content for summarization requests."""
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["empty_content"])

    return content.strip()
