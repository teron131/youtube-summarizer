"""
This module is responsible for handling audio transcription. It includes functions for optimizing audio files for transcription and for interfacing with the FAL transcription service.
"""

import io
import os

import fal_client
from pydub import AudioSegment

from .utils import log_and_print, whisper_result_to_txt


def optimize_audio_for_transcription(audio_bytes: bytes) -> bytes:
    """
    Optimizes audio for transcription using standard MP3 quality.
    Converts to mono and applies minimal compression to maintain quality.
    """
    raw_size_mb = len(audio_bytes) / 1024 / 1024
    log_and_print(f"🔄 Optimizing audio ({raw_size_mb:.1f}MB) for transcription...")

    try:
        audio_io = io.BytesIO(audio_bytes)
        audio_segment = None

        # Try loading with yt-dlp's preferred formats first (m4a, mp4)
        for fmt in ["m4a", "mp4", "mp3", "webm", "ogg"]:
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io, format=fmt)
                log_and_print(f"✅ Loaded audio source as {fmt}")
                break
            except Exception as e:
                log_and_print(f"❌ Failed to load as {fmt}: {e}")
                continue

        # Fallback to auto-detection if specific formats fail
        if not audio_segment:
            audio_io.seek(0)
            audio_segment = AudioSegment.from_file(audio_io)
            log_and_print("✅ Loaded audio source with auto-detection")

        # Get original audio properties
        original_channels = audio_segment.channels
        original_duration = len(audio_segment) / 1000.0  # Convert to seconds

        log_and_print(f"📊 Original audio: {original_channels} channels, {original_duration:.1f}s")

        # Validate duration - if too short, likely a parsing error
        expected_duration = raw_size_mb * 60 / 2  # Rough estimate: ~2MB per minute for m4a
        if original_duration < 10 or original_duration < expected_duration * 0.1:
            log_and_print(f"⚠️ Audio duration seems incorrect ({original_duration:.1f}s vs expected ~{expected_duration:.1f}s)")
            log_and_print(f"🔄 Skipping optimization due to duration mismatch, using original audio")
            return audio_bytes

        # Simple optimization: just convert to mono if needed, minimal compression
        if original_channels > 1:
            audio_segment = audio_segment.set_channels(1)
            log_and_print("🔄 Converted to mono")
        else:
            log_and_print("✅ Already mono")

        # Export with standard MP3 quality (128kbps - standard quality)
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format="mp3", bitrate="128k")
        compressed_bytes = output_buffer.getvalue()

        compressed_size_mb = len(compressed_bytes) / 1024 / 1024
        log_and_print(f"✅ Export complete. New size: {compressed_size_mb:.1f}MB")

        # Validate the compressed audio isn't too small
        if compressed_size_mb < 0.1 and raw_size_mb > 1:
            log_and_print(f"⚠️ Compressed audio is suspiciously small ({compressed_size_mb:.1f}MB)")
            log_and_print(f"🔄 Using original audio instead")
            return audio_bytes

        return compressed_bytes

    except Exception as e:
        log_and_print(f"❌ Audio optimization failed: {e}")
        log_and_print(f"⚠️ Could not process audio. Using original audio file ({raw_size_mb:.1f}MB).")
        return audio_bytes


def transcribe_with_fal(audio_bytes: bytes) -> str:
    """
    Transcribes audio using the FAL API.
    Handles audio upload, transcription job submission, and result formatting.
    """
    try:
        log_and_print("🎤 Starting FAL transcription...")
        if not os.getenv("FAL_KEY"):
            return "[FAL_KEY not configured]"

        # Log audio details for debugging
        audio_size_mb = len(audio_bytes) / 1024 / 1024
        log_and_print(f"📊 Audio size for transcription: {audio_size_mb:.2f}MB")

        if audio_size_mb < 0.001:
            log_and_print("⚠️ Warning: Audio file is extremely small, transcription quality may be poor")

        log_and_print("📤 Uploading audio to FAL...")
        # FAL is robust; a generic 'audio/mpeg' is sufficient for MP3.
        url = fal_client.upload(data=audio_bytes, content_type="audio/mpeg")
        log_and_print("✅ Upload successful to FAL")

        log_and_print("🔄 Starting transcription job...")

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log_entry in update.logs:
                    log_and_print(f"FAL: {log_entry['message']}")

        result = fal_client.subscribe(
            "fal-ai/whisper",
            arguments={
                "audio_url": url,
                "task": "transcribe",
                "language": None,
                # Add Whisper parameters for better transcription
                "model": "base",  # Use base model for faster processing
                "word_timestamps": False,  # Disable word timestamps for faster processing
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Log transcription result details
        if hasattr(result, "get"):
            transcript_text = whisper_result_to_txt(result)
            log_and_print(f"📝 Transcription result: {len(transcript_text)} characters")
            if len(transcript_text) < 100:
                log_and_print("⚠️ Warning: Transcription result is very short, audio quality may be poor")
        else:
            transcript_text = str(result)
            log_and_print(f"📝 Raw transcription result type: {type(result)}")

        log_and_print("✅ Transcription completed")
        return whisper_result_to_txt(result)

    except Exception as e:
        error_msg = str(e)
        log_and_print(f"❌ FAL transcription failed: {error_msg}")
        if "403" in error_msg or "forbidden" in error_msg.lower():
            return "[FAL API access denied (403). Check API key permissions.]"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "[FAL API quota exceeded]"
        elif "timeout" in error_msg.lower():
            return "[FAL API timeout - audio may be too long]"
        else:
            return f"[FAL transcription failed: {error_msg}]"
