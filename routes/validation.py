"""Request validation utilities"""

from fastapi import HTTPException
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url


def validate_url(url: str) -> str:
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    url = url.strip()
    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    return clean_youtube_url(url)


def validate_content(content: str) -> str:
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    return content.strip()
