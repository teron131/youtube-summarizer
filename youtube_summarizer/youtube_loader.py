"""
YouTube Video and Audio Loader
----------------------

This module provides robust functions to extract information and content from YouTube videos, including video details, subtitles, and audio streams. It uses `yt-dlp` with multiple strategies to ensure reliable and efficient data extraction, especially in cloud environments like Railway.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yt_dlp

from .utils import log_and_print, parse_youtube_json_captions, s2hk, srt_to_txt

logger = logging.getLogger(__name__)


def extract_browser_cookies() -> Optional[str]:
    """
    Extract cookies from browser for YouTube authentication.
    Returns path to cookies file or None if extraction fails.
    
    TEMPORARILY DISABLED due to dependency conflicts with curl_cffi/eventlet/trio
    """
    log_and_print("🍪 Cookie extraction temporarily disabled due to dependency conflicts")
    return None


def get_enhanced_client_configs():
    """
    Get enhanced client configurations with latest headers and better anti-detection.
    """
    return [
        # 1. Latest Chrome Desktop with realistic headers
        {
            "name": "Chrome Desktop (Latest)",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            },
            "extractor_args": {
                "youtube": {
                    "player_client": ["web", "web_creator"],
                    "player_skip": ["webpage", "configs"],
                }
            },
        },
        # 2. Firefox Desktop
        {
            "name": "Firefox Desktop",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
            "extractor_args": {
                "youtube": {
                    "player_client": ["web"],
                }
            },
        },
        # 3. Android Chrome Mobile
        {
            "name": "Android Mobile Chrome",
            "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "sec-ch-ua-mobile": "?1",
                "sec-ch-ua-platform": '"Android"',
            },
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"],
                }
            },
        },
        # 4. iOS Safari Mobile
        {
            "name": "iOS Safari Mobile",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
            "http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
            "extractor_args": {
                "youtube": {
                    "player_client": ["ios", "web"],
                }
            },
        },
        # 5. TV Client (often bypasses restrictions)
        {
            "name": "Smart TV Client",
            "user_agent": "Mozilla/5.0 (SMART-TV; LINUX; Tizen 6.0) AppleWebKit/537.36 (KHTML, like Gecko) Version/6.0 TV Safari/537.36",
            "http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
            "extractor_args": {
                "youtube": {
                    "player_client": ["tv_embedded", "web"],
                }
            },
        },
    ]


def extract_video_info_with_cookies(url: str, cookies_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract video information with cookie authentication support.
    """
    client_configs = get_enhanced_client_configs()

    for i, client_config in enumerate(client_configs):
        try:
            log_and_print(f"🔄 Attempt {i+1}/{len(client_configs)}: Using {client_config['name']} client...")

            # Base yt-dlp options
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "forcejson": True,
                "socket_timeout": 90,
                "user_agent": client_config["user_agent"],
                "referer": "https://www.youtube.com/",
                "http_headers": client_config["http_headers"],
                # Enhanced anti-detection
                "nocheckcertificate": True,
                "ignoreerrors": False,
                "no_color": True,
                "retries": 3,
                "fragment_retries": 3,
                "extractor_retries": 3,
                # Format selection
                "format": "bestaudio/best",
                "prefer_ffmpeg": True,
                "keepvideo": False,
                # Advanced anti-detection
                "sleep_interval": 1,
                "max_sleep_interval": 5,
                "sleep_interval_requests": 1,
                "sleep_interval_subtitles": 1,
                # Geographic and network settings
                "geo_bypass": True,
                "geo_bypass_country": "US",
                # Client-specific extractor args
                "extractor_args": client_config.get("extractor_args", {}),
            }

            # Add cookies if available
            if cookies_file and Path(cookies_file).exists():
                ydl_opts["cookiefile"] = cookies_file
                log_and_print(f"🍪 Using cookies from: {cookies_file}")

            # Random delay to avoid rate limiting
            time.sleep(1)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url, download=False)
                log_and_print(f"✅ Success with {client_config['name']} client!")
                return result

        except Exception as e:
            error_msg = str(e)
            log_and_print(f"❌ {client_config['name']} client failed: {error_msg}")

            # Add delays between failures to avoid aggressive rate limiting
            time.sleep(2)
            continue

    raise RuntimeError("All enhanced client configurations failed. YouTube's anti-bot measures are too aggressive.")


def extract_video_info_alternative_methods(url: str) -> Dict[str, Any]:
    """
    Alternative extraction methods as last resort.
    """
    log_and_print("🔄 Trying alternative extraction methods...")

    # Method 1: Use embedded player
    try:
        log_and_print("📺 Trying embedded player method...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "extractor_args": {
                "youtube": {
                    "player_client": ["tv_embedded"],
                    "player_skip": ["webpage"],
                }
            },
            "user_agent": "Mozilla/5.0 (SMART-TV; LINUX; Tizen 6.0) AppleWebKit/537.36",
            "referer": "https://www.youtube.com/",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            log_and_print("✅ Embedded player method succeeded!")
            return result

    except Exception as e:
        log_and_print(f"❌ Embedded player method failed: {str(e)}")

    # Method 2: Use age-gate bypass
    try:
        log_and_print("🔓 Trying age-gate bypass method...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "extractor_args": {
                "youtube": {
                    "player_client": ["web_creator", "web"],
                    "skip": ["hls", "dash"],
                }
            },
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "age_limit": 999,  # Bypass age restrictions
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            log_and_print("✅ Age-gate bypass method succeeded!")
            return result

    except Exception as e:
        log_and_print(f"❌ Age-gate bypass method failed: {str(e)}")

    raise RuntimeError("All alternative extraction methods failed.")


def extract_video_info(url: str) -> Dict[str, Any]:
    """
    Extract video information using multiple robust strategies with enhanced anti-detection.

    This function implements a multi-tier approach:
    1. Try cookie-based authentication with enhanced clients
    2. Fall back to alternative extraction methods
    3. Provide detailed error reporting for debugging
    """
    log_and_print("📋 Step 1: Extracting video info...")

    # First, try to extract browser cookies for authentication
    cookies_file = extract_browser_cookies()

    try:
        # Primary method: Enhanced clients with optional cookies
        return extract_video_info_with_cookies(url, cookies_file)

    except Exception as primary_error:
        log_and_print(f"❌ Primary extraction methods failed: {str(primary_error)}")

        try:
            # Fallback method: Alternative extraction strategies
            return extract_video_info_alternative_methods(url)

        except Exception as fallback_error:
            # Final error with comprehensive information
            error_msg = f"""
YouTube extraction completely failed. This may be due to:
1. YouTube's aggressive bot detection
2. Geographic restrictions
3. Video privacy settings
4. Network connectivity issues

Primary error: {str(primary_error)}
Fallback error: {str(fallback_error)}

Suggestions:
- Try again in a few minutes
- Use a different network/VPN
- Try a different video
- Check if the video is available in your region
"""
            log_and_print(f"💔 Complete extraction failure: {error_msg}")
            raise RuntimeError(error_msg.strip())

    finally:
        # Clean up cookies file if we created one
        if cookies_file and Path(cookies_file).exists():
            try:
                Path(cookies_file).unlink()
                log_and_print("🧹 Cleaned up temporary cookies file")
            except Exception:
                pass


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
