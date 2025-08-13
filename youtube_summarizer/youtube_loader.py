"""
YouTube Video and Audio Loader
------------------------------

This module provides functions to extract information and content from YouTube videos, including video details, subtitles, and audio streams. It uses `yt-dlp` with multiple strategies to ensure robust and reliable data extraction, especially in cloud environments like Railway.
"""

import logging
from typing import Any, Dict

import requests
import yt_dlp

from .utils import log_and_print, parse_youtube_json_captions, s2hk, srt_to_txt

logger = logging.getLogger(__name__)


def extract_video_info(url: str) -> Dict[str, Any]:
    """
    Extract video information using yt-dlp with a robust, unified strategy.

    This function uses a single, reliable configuration for yt-dlp to fetch
    video metadata, ensuring consistency across different environments. It includes
    error handling to gracefully manage extraction failures.
    """
    log_and_print(f"📡 Extracting video info for URL: {url}")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "forcejson": True,
        "socket_timeout": 30,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "http_headers": {
            "Accept-Language": "en-us,en;q=0.5",
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            log_and_print("✅ Video info extracted successfully!")
            return info
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "private video" in error_msg.lower() or "video unavailable" in error_msg.lower():
            raise RuntimeError("Video is private or unavailable.")
        raise RuntimeError(f"Failed to extract video information: {error_msg}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during video info extraction: {e}")


def get_subtitle_from_captions(info: Dict[str, Any]) -> str | None:
    """
    Extracts and formats existing subtitles from video info.

    Prioritizes Chinese (Traditional/Simplified) and English subtitles.
    """
    subtitles = info.get("subtitles", {})
    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in subtitles:
            subtitle_info = subtitles[lang][0]
            response = requests.get(subtitle_info["url"])
            raw_content = response.text

            if lang == "zh-CN":
                raw_content = s2hk(raw_content)

            if raw_content.strip().startswith("{"):
                return parse_youtube_json_captions(raw_content)
            else:
                return srt_to_txt(raw_content)
    return None


def download_audio_bytes(info: Dict[str, Any]) -> bytes:
    """
    Downloads audio from YouTube video info with optimized format selection for size.

    Selects the most efficient audio-only format (Opus, AAC) to minimize download size
    and processing time.
    """
    formats = info.get("formats", [])
    high_efficiency_formats = []
    medium_efficiency_formats = []
    fallback_formats = []

    for fmt in formats:
        if fmt.get("vcodec") == "none" and fmt.get("acodec") != "none":
            acodec = fmt.get("acodec", "").lower()
            filesize = fmt.get("filesize", 0) or fmt.get("filesize_approx", 0) or 0
            if any(codec in acodec for codec in ["opus"]):
                high_efficiency_formats.append((fmt, filesize))
            elif any(codec in acodec for codec in ["aac", "mp4a"]):
                medium_efficiency_formats.append((fmt, filesize))
            else:
                fallback_formats.append((fmt, filesize))
        elif fmt.get("acodec") != "none":
            filesize = fmt.get("filesize", 0) or fmt.get("filesize_approx", 0) or 0
            fallback_formats.append((fmt, filesize))

    high_efficiency_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))
    medium_efficiency_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))
    fallback_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))

    audio_format = None
    selected_from = "unknown"

    for fmt_list, category in [
        (high_efficiency_formats, "high_efficiency"),
        (medium_efficiency_formats, "medium_efficiency"),
        (fallback_formats, "fallback"),
    ]:
        if fmt_list:
            audio_format, _ = fmt_list[0]
            selected_from = category
            break

    if not audio_format:
        all_formats = [fmt for fmt_list in [high_efficiency_formats, medium_efficiency_formats, fallback_formats] for fmt, _ in fmt_list]
        if all_formats:
            audio_format = all_formats[0]
            selected_from = "any_available"
        else:
            raise RuntimeError("No audio format available")

    filesize_mb = (audio_format.get("filesize") or audio_format.get("filesize_approx") or 0) / 1024 / 1024
    log_and_print(f"Selected format: {audio_format.get('format_id')} ({selected_from}) - {audio_format.get('acodec')} - {filesize_mb:.1f}MB estimated")

    audio_url = audio_format["url"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.youtube.com/",
    }

    try:
        response = requests.get(audio_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if "403" in str(e):
            raise RuntimeError("YouTube blocked audio download (HTTP 403). Try again later or use a different video.")
        else:
            raise RuntimeError(f"Failed to download audio: {e}")

    audio_data = b""
    for chunk in response.iter_content(chunk_size=32768):
        if chunk:
            audio_data += chunk

    log_and_print(f"Downloaded {len(audio_data)} bytes of audio")
    return audio_data
