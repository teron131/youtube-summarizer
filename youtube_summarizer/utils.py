"""
Utility Functions
-----------------

This module contains helper functions for string manipulation, data parsing, and logging, used across the YouTube Summarizer application.
"""

import json
import re
import sys

from opencc import OpenCC


def log_and_print(message: str):
    """Log and print message to ensure visibility in Railway."""
    print(message, flush=True)
    sys.stdout.flush()


def is_youtube_url(url: str) -> bool:
    """
    Check if the URL is a valid YouTube URL.
    Accepts both youtube.com/watch?v= and youtu.be/ formats.
    """
    youtube_patterns = [
        r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
        r"https?://(?:www\.)?youtu\.be/[\w-]+",
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def clean_youtube_url(url: str) -> str:
    """
    Clean the YouTube URL by extracting video ID and removing extra parameters.
    Converts both formats to standard youtube.com/watch?v=ID format.
    """
    # Extract video ID from youtube.com/watch?v=ID format
    youtube_match = re.search(r"youtube\.com/watch\?v=([\w-]+)", url)
    if youtube_match:
        video_id = youtube_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    # Extract video ID from youtu.be/ID format
    youtu_be_match = re.search(r"youtu\.be/([\w-]+)", url)
    if youtu_be_match:
        video_id = youtu_be_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    # Return original URL if no match found
    return url


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Hong Kong variant)."""
    return OpenCC("s2hk").convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    txt_content = "\n".join(chunk["text"].strip() for chunk in result.get("chunks", []))
    return s2hk(txt_content)


def parse_youtube_json_captions(json_content: str) -> str:
    """
    Parse YouTube's JSON timedtext format and extract plain text.
    Handles the specific structure of YouTube's auto-generated captions.
    """
    try:
        data = json.loads(json_content)
        text_parts = []
        if "events" in data:
            for event in data["events"]:
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            text_parts.append(seg["utf8"])
        full_text = "".join(text_parts)
        return full_text.strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, return the original content
        return json_content


def srt_to_txt(srt_content: str) -> str:
    """Convert SRT (SubRip Text) format content to plain text."""
    lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        # Filter out timestamp lines and sequence numbers
        if line and not line.isdigit() and "-->" not in line:
            lines.append(line)
    return s2hk("\n".join(lines))
