"""Top-level package exports for the YouTube summarizer."""

from .schemas import Summary
from .scrapper import extract_transcript_text, has_transcript_provider_key
from .settings import AppSettings, get_settings
from .summarizer_gemini import summarize_video_async as summarize_video_gemini
from .summarizer_openrouter import summarize_video_async as summarize_video_openrouter
from .utils import clean_youtube_url, is_youtube_url

__all__ = [
    "AppSettings",
    "Summary",
    "clean_youtube_url",
    "extract_transcript_text",
    "get_settings",
    "has_transcript_provider_key",
    "is_youtube_url",
    "summarize_video_gemini",
    "summarize_video_openrouter",
]
