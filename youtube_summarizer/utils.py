import re

from opencc import OpenCC


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing.

    Args:
        text: The text to clean

    Returns:
        Cleaned text string
    """
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", r"\n\n", text)
    # Remove excessive spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def clean_youtube_url(url: str) -> str:
    """Clean and normalize a YouTube URL.

    Args:
        url: YouTube URL in various formats

    Returns:
        Cleaned YouTube URL
    """
    # Remove query parameters except v
    if "youtube.com/watch" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    elif "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL.

    Args:
        url: URL to check

    Returns:
        True if URL is a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r"youtube\.com/watch\?v=",
        r"youtu\.be/",
        r"youtube\.com/embed/",
        r"youtube\.com/v/",
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    return OpenCC("s2hk").convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    chunks = result.get("chunks") or []
    lines = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text")]
    return s2hk("\n".join(lines))
