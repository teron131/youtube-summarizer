"""
YouTube Video and Audio Loader - Simplified Hybrid Implementation
----------------------------------------------------------------

Simple hybrid approach:
1. Use pytubefix to check for existing captions
2. If captions exist, load them as text
3. Otherwise, use yt-dlp to download audio for transcription
"""

import os

import requests
import yt_dlp
from dotenv import load_dotenv
from pytubefix import YouTube

from .transcriber import optimize_audio_for_transcription, transcribe_with_fal

load_dotenv()


def download_audio_with_ytdlp(url: str) -> bytes:
    """Download audio using yt-dlp with robust configuration."""

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "referer": "https://www.youtube.com/",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Find best audio format
            formats = info.get("formats", [])
            audio_format = None

            for fmt in formats:
                if fmt.get("vcodec") == "none" and fmt.get("acodec") != "none":
                    audio_format = fmt
                    break

            if not audio_format:
                # Fallback to any format with audio
                for fmt in formats:
                    if fmt.get("acodec") != "none":
                        audio_format = fmt
                        break

            if not audio_format:
                raise RuntimeError("No audio format found")

            # Download audio
            audio_url = audio_format["url"]
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.youtube.com/",
            }

            response = requests.get(audio_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            audio_data = b""
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    audio_data += chunk

            print(f"Downloaded {len(audio_data)} bytes of audio")
            return audio_data

    except Exception as e:
        raise RuntimeError(f"yt-dlp audio download failed: {e}")


def youtube_loader(url: str) -> str:
    """
    Load and process YouTube video using hybrid approach.

    Args:
        url: YouTube video URL

    Returns:
        Formatted string with video info and subtitle
    """
    print(f"\n🎬 Loading YouTube video: {url}")
    print("=" * 50)

    try:
        # Step 1: Use pytubefix for metadata and caption detection
        print("📋 Fetching metadata and checking captions with pytubefix...")
        youtube = YouTube(url, client="WEB")

        title = youtube.title
        author = youtube.author
        duration = getattr(youtube, "length", "Unknown")

        print(f"✅ Video accessible:")
        print(f"   📺 Title: {title}")
        print(f"   👤 Author: {author}")
        print(f"   ⏱️  Duration: {duration}s")

        # Step 2: Check for available captions
        print("🔍 Checking for available captions...")
        available_captions = list(youtube.captions.keys())
        print(f"📝 Available captions: {available_captions}")

        subtitle = None

        if available_captions:
            # Priority order for captions
            caption_priorities = ["zh-HK", "zh-CN", "en", "a.en"]

            for caption_obj in available_captions:
                if caption_obj.code in caption_priorities:
                    try:
                        print(f"✅ Attempting to use caption: {caption_obj.code} ({caption_obj.name})")
                        print(f"🔄 Generating txt captions for {caption_obj.code}...")
                        caption_text = caption_obj.generate_txt_captions()

                        if caption_text and caption_text.strip():
                            print(f"📖 Successfully loaded caption {caption_obj.code}, length: {len(caption_text)} characters")
                            subtitle = caption_text
                            break
                        else:
                            print(f"⚠️ Caption {caption_obj.code} generated empty text, trying next...")
                    except Exception as e:
                        print(f"❌ Failed to load caption {caption_obj.code}: {e}")
                        print(f"   Error type: {type(e).__name__}")
                        continue

            if not subtitle:
                print("❌ All caption loading attempts failed, falling back to audio transcription...")

        # Step 3: If no captions, use yt-dlp for audio transcription
        if not subtitle:
            if not available_captions:
                print("🎵 No captions available, downloading audio for transcription...")
            else:
                print("🎵 Caption loading failed, downloading audio for transcription...")

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
