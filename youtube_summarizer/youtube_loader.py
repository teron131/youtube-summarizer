"""
YouTube Video and Audio Loader - yt-dlp only
----------------------------------------------------------------

Optimized yt-dlp approach:
1. Use yt-dlp to get video metadata, including captions
2. If captions exist, load them as text
3. Otherwise, use yt-dlp to download audio for transcription
"""

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp
from dotenv import load_dotenv

from .transcriber import optimize_audio_for_transcription, transcribe_with_fal

load_dotenv()

# Constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
YOUTUBE_REFERER = "https://www.youtube.com/"
SUBTITLE_LANGUAGES = ["en", "a.en", "zh-HK", "zh-CN"]
SUBTITLE_FILTER_PATTERNS = ["-->", "WEBVTT", "NOTE"]

# Configure logging
logger = logging.getLogger(__name__)

# Base yt-dlp configuration
BASE_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "user_agent": USER_AGENT,
    "referer": YOUTUBE_REFERER,
}

# Metadata extraction configuration
METADATA_YDL_OPTS = {
    **BASE_YDL_OPTS,
    "subtitleslangs": SUBTITLE_LANGUAGES,
    "writesubtitles": True,
    "skip_download": True,
}

# Audio download configuration
AUDIO_YDL_OPTS = {
    **BASE_YDL_OPTS,
    "format": "bestaudio[ext=m4a]/bestaudio/best",
}


def get_best_thumbnail(thumbnails: List[Dict[str, Any]]) -> Optional[str]:
    """Select the best thumbnail from a list, prioritizing resolution."""
    if not thumbnails:
        return None

    best_thumbnail = max(thumbnails, key=lambda t: t.get("height", 0))
    return best_thumbnail.get("url")


@contextmanager
def temp_file_manager(prefix: str = "youtube_", suffix: str = ".%(ext)s"):
    """Context manager for temporary file handling."""
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = temp_dir / f"{prefix}{os.getpid()}{suffix}"

    try:
        yield str(temp_filename)
    finally:
        # Clean up any files that match the pattern
        if suffix == ".%(ext)s":
            # Handle yt-dlp template - find actual files
            pattern = f"{prefix}{os.getpid()}.*"
            for file_path in temp_dir.glob(pattern):
                try:
                    file_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {file_path}: {e}")
        else:
            try:
                temp_filename.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_filename}: {e}")


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
    """Download audio using yt-dlp with robust configuration."""
    logger.info(f"Downloading audio for: {url}")

    with temp_file_manager("youtube_audio_") as temp_filename:
        # Ensure explicit download with skip_download: False
        opts = {**AUDIO_YDL_OPTS, "outtmpl": temp_filename, "skip_download": False}  # Explicitly enable download

        try:
            result = _extract_yt_dlp_info(url, opts)
            logger.debug(f"yt-dlp result keys: {list(result.keys()) if result else 'None'}")

            # Method 1: Try to get filepath from requested_downloads
            downloaded_file = None
            if result.get("requested_downloads"):
                downloaded_file = result["requested_downloads"][0].get("filepath")
                logger.debug(f"Filepath from requested_downloads: {downloaded_file}")

            # Method 2: If that fails, look for files matching our pattern
            if not downloaded_file or not os.path.exists(downloaded_file):
                logger.debug("requested_downloads method failed, searching for downloaded files...")

                # Look for files that match our pattern
                temp_dir = Path(tempfile.gettempdir())
                pattern = f"youtube_audio_{os.getpid()}.*"
                matching_files = list(temp_dir.glob(pattern))

                logger.debug(f"Found {len(matching_files)} files matching pattern: {pattern}")
                for file_path in matching_files:
                    logger.debug(f"  - {file_path}")

                if matching_files:
                    # Use the first (and should be only) matching file
                    downloaded_file = str(matching_files[0])
                    logger.debug(f"Using pattern-matched file: {downloaded_file}")

            # Method 3: If still no file, check if the template filename exists as-is
            if not downloaded_file or not os.path.exists(downloaded_file):
                template_file = temp_filename.replace(".%(ext)s", ".m4a")  # Common extension
                if os.path.exists(template_file):
                    downloaded_file = template_file
                    logger.debug(f"Using template-based file: {downloaded_file}")

            # Final check
            if not downloaded_file or not os.path.exists(downloaded_file):
                # List all temp files for debugging
                temp_dir = Path(tempfile.gettempdir())
                all_temp_files = [f for f in temp_dir.iterdir() if f.name.startswith("youtube_")]
                logger.error(f"No downloaded file found. Temp files in directory: {[str(f) for f in all_temp_files[:10]]}")  # Limit to 10 for logging
                raise RuntimeError("Downloaded audio file not found after multiple detection methods")

            # Read the audio data
            logger.info(f"Reading audio data from: {downloaded_file}")
            audio_data = Path(downloaded_file).read_bytes()

            # Clean up the downloaded file
            try:
                Path(downloaded_file).unlink(missing_ok=True)
                logger.debug(f"Cleaned up downloaded file: {downloaded_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up downloaded file {downloaded_file}: {e}")

            logger.info(f"Downloaded {len(audio_data)} bytes of audio")
            return audio_data

        except Exception as e:
            # Enhanced error logging
            logger.error(f"Audio download failed: {e}")
            logger.error(f"URL: {url}")
            logger.error(f"Template: {temp_filename}")

            # List temp directory contents for debugging
            try:
                temp_dir = Path(tempfile.gettempdir())
                recent_files = [f for f in temp_dir.iterdir() if f.name.startswith("youtube_")][:5]
                logger.error(f"Recent temp files: {[str(f) for f in recent_files]}")
            except Exception:
                pass

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
