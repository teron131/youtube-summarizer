"""
YouTube Video and Audio Loader - yt-dlp only
----------------------------------------------------------------

Optimized yt-dlp approach:
1. Use yt-dlp to get video metadata, including captions
2. If captions exist, load them as text
3. Otherwise, use yt-dlp to download audio for transcaription
"""

import logging
import os
from typing import Any, Dict, List, Optional

import yt_dlp
from dotenv import load_dotenv

from .transcriber import optimize_audio_for_transcription, transcribe_with_fal

load_dotenv()

# Constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
YOUTUBE_REFERER = "https://www.youtube.com/"
SUBTITLE_LANGUAGES = ["en", "a.en", "zh-HK", "zh-CN"]
SUBTITLE_FILTER_PATTERNS = ["-->", "WEBVTT", "NOTE"]

# Configure logging
logger = logging.getLogger(__name__)

# Base yt-dlp configuration with enhanced headers and options
BASE_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "user_agent": USER_AGENT,
    "referer": YOUTUBE_REFERER,
    "retries": 3,
    "fragment_retries": 3,
    "extractor_retries": 3,
    "http_headers": {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-us,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
}

# Metadata extraction configuration
METADATA_YDL_OPTS = {
    **BASE_YDL_OPTS,
    "subtitleslangs": SUBTITLE_LANGUAGES,
    "writesubtitles": True,
    "skip_download": True,
}

# Audio download configuration with format fallbacks
AUDIO_YDL_OPTS = {
    **BASE_YDL_OPTS,
    # Use specific audio format IDs and fallbacks for m3u8 streams
    "format": "234/233/bestaudio[protocol*=m3u8]/bestaudio[ext=m4a]/bestaudio/best[height<=480]",
    "extractaudio": False,  # Don't re-extract since we're getting audio-only formats
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "128",
        }
    ],
}


def get_best_thumbnail(thumbnails: List[Dict[str, Any]]) -> Optional[str]:
    """Select the best thumbnail from a list, prioritizing resolution."""
    if not thumbnails:
        return None

    best_thumbnail = max(thumbnails, key=lambda t: t.get("height", 0))
    return best_thumbnail.get("url")


def _extract_yt_dlp_info(url: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    """Extract video information using yt-dlp with given options."""
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=opts.get("skip_download", True) is False)
    except Exception as e:
        raise RuntimeError(f"yt-dlp extraction failed: {e}") from e


def _process_subtitle_file(filepath: str) -> Optional[str]:
    """Process subtitle file and extract clean text."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Skip timestamp lines, numbers, and VTT headers
                if any(pattern in line for pattern in SUBTITLE_FILTER_PATTERNS) or line.isdigit() or line.startswith("STYLE") or line.startswith("::cue"):
                    continue

                lines.append(line)

            subtitle_text = "\n".join(lines)
            return subtitle_text if subtitle_text.strip() else None

    except Exception as e:
        logger.warning(f"Failed to process subtitle file {filepath}: {e}")
        return None


def extract_video_info(url: str) -> Dict[str, Any]:
    """
    Extract basic video information using yt-dlp.

    Args:
        url: YouTube video URL (should be cleaned/validated before calling)

    Returns:
        Dictionary containing video metadata

    Raises:
        RuntimeError: If video info extraction fails
    """
    logger.info(f"Extracting video info for: {url}")

    try:
        info = _extract_yt_dlp_info(url, {"quiet": True, "no_warnings": True})

        # Select the best thumbnail URL
        thumbnail_url = get_best_thumbnail(info.get("thumbnails", [])) or info.get("thumbnail")

        duration = info.get("duration", 0)
        metadata = {
            "title": info.get("title"),
            "author": info.get("uploader"),
            "duration": f"{duration}s" if duration else None,
            "duration_seconds": duration,
            "thumbnail": thumbnail_url,
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "url": url,
        }

        logger.info(f"Video info extracted: {metadata['title']} by {metadata['author']}")
        return metadata

    except Exception as e:
        error_msg = f"Failed to extract video info: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def download_audio(url: str) -> bytes:
    """Download audio using yt-dlp directly into memory."""
    import glob
    import os
    import tempfile

    logger.info(f"Downloading audio for: {url}")

    # Simplified and more reliable configuration
    download_opts = {
        **BASE_YDL_OPTS,
        # Use best audio format available, prefer m4a/mp3 for compatibility
        "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
        # Don't use postprocessors - they can cause file naming issues
        "extractaudio": False,
        # Simple output template
        "outtmpl": "audio.%(ext)s",
        # Keep original format to avoid conversion issues
        "keepvideo": False,
    }

    try:
        # Create a temporary directory for the download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for download
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                logger.info("Starting yt-dlp download...")

                # Download the audio file
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                if not info:
                    raise RuntimeError("Failed to extract video info")

                logger.info("Download completed, searching for audio files...")

                # List all files in the temp directory for debugging
                all_files = os.listdir(".")
                logger.info(f"Files in temp directory: {all_files}")

                # Look for any audio files using glob patterns
                audio_patterns = ["*.m4a", "*.mp3", "*.webm", "*.opus", "*.aac", "*.wav"]
                downloaded_file = None

                for pattern in audio_patterns:
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        downloaded_file = matching_files[0]  # Take the first match
                        logger.info(f"Found audio file: {downloaded_file}")
                        break

                # Fallback: look for files with 'audio' in the name
                if not downloaded_file:
                    for filename in all_files:
                        if any(ext in filename.lower() for ext in [".m4a", ".mp3", ".webm", ".opus", ".aac", ".wav"]):
                            downloaded_file = filename
                            logger.info(f"Found audio file (fallback): {downloaded_file}")
                            break

                if not downloaded_file:
                    # Final fallback: take the largest file (likely the audio)
                    if all_files:
                        largest_file = max(all_files, key=lambda f: os.path.getsize(f) if os.path.isfile(f) else 0)
                        if os.path.getsize(largest_file) > 1024:  # At least 1KB
                            downloaded_file = largest_file
                            logger.info(f"Using largest file as audio: {downloaded_file}")

                if not downloaded_file:
                    raise RuntimeError(f"No audio file found after download. Files in directory: {all_files}")

                # Read the file into memory
                file_path = os.path.join(temp_dir, downloaded_file)
                logger.info(f"Reading audio file: {file_path}")

                with open(file_path, "rb") as f:
                    audio_data = f.read()

                if len(audio_data) == 0:
                    raise RuntimeError(f"Downloaded file is empty: {downloaded_file}")

                logger.info(f"Successfully downloaded {len(audio_data)} bytes of audio data")
                return audio_data

            finally:
                os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Audio download failed: {e}")
        raise RuntimeError(f"Audio download failed: {e}") from e


def _extract_captions(info: Dict[str, Any]) -> Optional[str]:
    """Extract and process captions from yt-dlp info."""
    if not info.get("requested_subtitles"):
        return None

    for lang, sub_info in info["requested_subtitles"].items():
        filepath = sub_info.get("filepath")
        if not filepath or not os.path.exists(filepath):
            continue

        logger.info(f"Processing caption file for {lang}: {filepath}")
        subtitle = _process_subtitle_file(filepath)

        # Clean up the subtitle file
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to clean up subtitle file {filepath}: {e}")

        if subtitle and subtitle.strip():
            logger.info(f"Successfully loaded caption {lang}, length: {len(subtitle)} characters")
            return subtitle

    return None


def youtube_loader(url: str) -> str:
    """
    Load and process YouTube video using an optimized yt-dlp approach.

    Args:
        url: YouTube video URL

    Returns:
        Formatted string with video info and subtitle

    Raises:
        RuntimeError: If video loading fails
    """
    logger.info(f"Loading YouTube video: {url}")

    try:
        # Step 1: Extract metadata and captions
        logger.info("Fetching metadata and checking captions...")
        info = _extract_yt_dlp_info(url, METADATA_YDL_OPTS)

        title = info.get("title", "Unknown Title")
        author = info.get("uploader", "Unknown Author")
        duration = info.get("duration", "Unknown")

        logger.info(f"Video accessible - Title: {title}, Author: {author}, Duration: {duration}s")

        # Step 2: Try to extract captions
        subtitle = _extract_captions(info)

        # Step 3: If no captions, download and transcribe audio
        if not subtitle:
            logger.info("No captions available, downloading audio for transcription...")

            fal_key = os.getenv("FAL_KEY")
            if not fal_key:
                raise RuntimeError("FAL_KEY not configured - please set your FAL API key")

            # Download and transcribe audio
            audio_bytes = download_audio(url)

            logger.info("Optimizing audio for transcription...")
            optimized_audio = optimize_audio_for_transcription(audio_bytes)

            logger.info("Transcribing audio...")
            subtitle = transcribe_with_fal(optimized_audio)

        # Step 4: Format final result
        content_parts = [
            "Answer the user's question based on the full content.",
            f"Title: {title}",
            f"Author: {author}",
            f"Subtitle:\n{subtitle}",
        ]

        logger.info("Video processed successfully!")
        return "\n".join(content_parts)

    except Exception as e:
        error_msg = f"Failed to load YouTube video: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
