"""
YouTube Video and Audio Loader
------------------------------

This module provides functions to extract information and content from YouTube videos,
including video details, subtitles, and audio streams. It uses `yt-dlp` with
multiple strategies to ensure robust and reliable data extraction, especially in
cloud environments like Railway.
"""

import json
import logging
import os
import random
from typing import Any, Dict

import requests
import yt_dlp
from .utils import log_and_print, s2hk, parse_youtube_json_captions, srt_to_txt

logger = logging.getLogger(__name__)


def extract_video_info(url: str) -> Dict[str, Any]:
    """
    Extract video information using yt-dlp with Railway-optimized strategies.
    Cycles through multiple user-agents and client configurations to avoid common
    HTTP errors.
    """
    is_cloud_env = any(
        [
            os.getenv("RAILWAY_STATIC_URL"),
            os.getenv("RAILWAY_PROJECT_ID"),
            os.getenv("PORT") and not os.getenv("DEVELOPMENT"),
            os.getenv("RENDER"),
            "/tmp" in os.getcwd(),
        ]
    )

    if is_cloud_env:
        log_and_print("🌐 Detected cloud environment - using optimized strategies")

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36",
    ]

    strategies = [
        {
            "user_agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip",
            "extractor_args": {"youtube": {"player_client": ["android_tv"], "player_skip": ["configs"], "skip": ["hls", "dash"]}},
        },
        {
            "user_agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip",
            "extractor_args": {"youtube": {"player_client": ["android"], "player_skip": ["configs"]}},
        },
        {
            "user_agent": "com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)",
            "extractor_args": {"youtube": {"player_client": ["ios"], "player_skip": ["configs"]}},
        },
        {
            "user_agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36",
            "extractor_args": {"youtube": {"player_client": ["android_embedded"], "player_skip": ["configs", "webpage"]}},
        },
        {
            "user_agent": random.choice(user_agents),
            "extractor_args": {"youtube": {"player_client": ["web"], "player_skip": ["configs"]}},
            "age_limit": 999,
        },
    ]

    last_error = None

    for i, strategy in enumerate(strategies):
        log_and_print(f"Trying strategy {i + 1}/{len(strategies)}...")
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "user_agent": strategy["user_agent"],
            "referer": "https://www.youtube.com/",
            "extractor_args": strategy["extractor_args"],
            "socket_timeout": 30,
            "http_headers": {
                "User-Agent": strategy["user_agent"],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Sec-Fetch-Mode": "navigate",
            },
        }

        if "age_limit" in strategy:
            ydl_opts["age_limit"] = strategy["age_limit"]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                log_and_print(f"✅ Strategy {i + 1} succeeded!")
                return info
        except Exception as e:
            error_msg = str(e)
            last_error = e
            log_and_print(f"❌ Strategy {i + 1} failed: {error_msg}")
            if any(keyword in error_msg.lower() for keyword in ["private video", "video unavailable"]):
                break
            continue

    if last_error:
        raise RuntimeError(f"Failed to access video: {str(last_error)}")
    raise RuntimeError("Unknown error occurred while processing YouTube video.")


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
