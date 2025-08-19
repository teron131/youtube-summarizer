"""
YouTube Video and Audio Loader - yt-dlp only
----------------------------------------------------------------

Simple yt-dlp approach:
1. Use yt-dlp to get video metadata, including captions
2. If captions exist, load them as text
3. Otherwise, use yt-dlp to download audio for transcription
"""

import os

import requests
import yt_dlp
from dotenv import load_dotenv

from .transcriber import optimize_audio_for_transcription, transcribe_with_fal

load_dotenv()

YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "format": "bestaudio/best",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "referer": "https://www.youtube.com/",
    "subtitleslangs": ["en", "a.en", "zh-HK", "zh-CN"],
    "writesubtitles": True,
    "skip_download": True,
}


def download_audio_with_ytdlp(url: str) -> bytes:
    """Download audio using yt-dlp with robust configuration."""

    # Prioritize m4a, but fall back to best available audio
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": "-",  # Output to stdout
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "referer": "https://www.youtube.com/",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # We are piping the content to stdout, so download=True is correct here
            result = ydl.extract_info(url, download=True)
            audio_data = result["requested_downloads"][0]["_final_filename"]

            print(f"Downloaded {len(audio_data)} bytes of audio")
            return audio_data

    except Exception as e:
        raise RuntimeError(f"yt-dlp audio download failed: {e}")


def youtube_loader(url: str) -> str:
    """
    Load and process YouTube video using a yt-dlp-only approach.

    Args:
        url: YouTube video URL

    Returns:
        Formatted string with video info and subtitle
    """
    print(f"\n🎬 Loading YouTube video: {url}")
    print("=" * 50)

    try:
        # Step 1: Use yt-dlp for metadata and caption detection
        print("📋 Fetching metadata and checking captions with yt-dlp...")
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown Title")
            author = info.get("uploader", "Unknown Author")
            duration = info.get("duration", "Unknown")

        print(f"✅ Video accessible:")
        print(f"   📺 Title: {title}")
        print(f"   👤 Author: {author}")
        print(f"   ⏱️  Duration: {duration}s")

        # Step 2: Check for available captions from requested_subtitles
        subtitle = None
        if info.get("requested_subtitles"):
            for lang, sub_info in info["requested_subtitles"].items():
                if sub_info.get("filepath") and os.path.exists(sub_info["filepath"]):
                    print(f"📖 Found downloaded caption file for {lang}: {sub_info['filepath']}")
                    with open(sub_info["filepath"], "r", encoding="utf-8") as f:
                        # Simple VTT/SRT to plain text conversion
                        lines = [line.strip() for line in f if "-->" not in line and not line.strip().isdigit() and line.strip()]
                        subtitle = "\n".join(lines)
                    # Clean up the subtitle file after reading
                    os.remove(sub_info["filepath"])
                    if subtitle and subtitle.strip():
                        print(f"✅ Successfully loaded caption {lang}, length: {len(subtitle)} characters")
                        break  # Use the first one found
                    else:
                        subtitle = None  # Reset if empty

        # Step 3: If no captions, use yt-dlp for audio transcription
        if not subtitle:
            print("🎵 No captions available, downloading audio for transcription...")

            if not os.getenv("FAL_KEY"):
                raise RuntimeError("FAL_KEY not configured - please set your FAL API key")

            # Download audio with yt-dlp
            audio_bytes = download_audio_with_ytdlp(url)

            # Optimize and transcribe
            print("🔧 Optimizing audio for transcription...")
            optimized_audio = optimize_audio_for_transcription(audio_bytes)

            print("🎤 Transcribing audio...")
            subtitle = transcribe_with_fal(optimized_audio)  # Already returns processed text

        # Step 4: Format final result
        content = [
            "Answer the user's question based on the full content.",
            f"Title: {title}",
            f"Author: {author}",
            f"subtitle:\n{subtitle}",
        ]

        print("✅ SUCCESS: Video processed successfully!")
        return "\n".join(content)

    except Exception as e:
        print(f"💥 FAILED: {e}")
        raise RuntimeError(f"Failed to load YouTube video: {e}")
