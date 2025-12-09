"""
This module is responsible for handling audio transcription. It includes functions for optimizing audio files for transcription and for interfacing with the FAL transcription service.
"""

import io
import logging
import os

import fal_client
from pydub import AudioSegment

from .utils import whisper_result_to_txt


def optimize_audio_for_transcription(audio_bytes: bytes) -> bytes:
    """
    Optimizes audio for transcription using standard MP3 quality.
    Converts to mono and applies minimal compression to maintain quality.
    """
    raw_size_mb = len(audio_bytes) / 1024 / 1024
    logging.info(f"🔄 Optimizing audio ({raw_size_mb:.1f}MB) for transcription...")

    try:
        audio_io = io.BytesIO(audio_bytes)
        audio_segment = None

        # Try loading with yt-dlp's preferred formats first (m4a, mp4)
        for fmt in ["m4a", "mp4", "mp3", "webm", "ogg"]:
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io, format=fmt)
                logging.info(f"✅ Loaded audio source as {fmt}")
                break
            except Exception as e:
                logging.warning(f"❌ Failed to load as {fmt}: {e}")
                continue

        # Fallback to auto-detection if specific formats fail
        if not audio_segment:
            audio_io.seek(0)
            audio_segment = AudioSegment.from_file(audio_io)
            logging.info("✅ Loaded audio source with auto-detection")

        # Get original audio properties
        original_channels = audio_segment.channels
        original_duration = len(audio_segment) / 1000.0  # Convert to seconds

        logging.info(f"📊 Original audio: {original_channels} channels, {original_duration:.1f}s")

        # Validate duration - if too short, likely a parsing error
        expected_duration = raw_size_mb * 60 / 2  # Rough estimate: ~2MB per minute for m4a
        if original_duration < 10 or original_duration < expected_duration * 0.1:
            logging.warning(f"⚠️ Audio duration seems incorrect ({original_duration:.1f}s vs expected ~{expected_duration:.1f}s)")
            logging.info("🔄 Skipping optimization due to duration mismatch, using original audio")
            return audio_bytes

        # Simple optimization: just convert to mono if needed, minimal compression
        if original_channels > 1:
            audio_segment = audio_segment.set_channels(1)
            logging.info("🔄 Converted to mono")
        else:
            logging.info("✅ Already mono")

        # Export with standard MP3 quality (128kbps - standard quality)
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format="mp3", bitrate="128k")
        compressed_bytes = output_buffer.getvalue()

        compressed_size_mb = len(compressed_bytes) / 1024 / 1024
        logging.info(f"✅ Export complete. New size: {compressed_size_mb:.1f}MB")

        # Validate the compressed audio isn't too small
        if compressed_size_mb < 0.1 and raw_size_mb > 1:
            logging.warning(f"⚠️ Compressed audio is suspiciously small ({compressed_size_mb:.1f}MB)")
            logging.info("🔄 Using original audio instead")
            return audio_bytes

        return compressed_bytes

    except Exception as e:
        logging.error(f"❌ Audio optimization failed: {e}")
        logging.info(f"⚠️ Could not process audio. Using original audio file ({raw_size_mb:.1f}MB).")
        return audio_bytes


def transcribe_with_fal(audio_bytes: bytes) -> str:
    """
    Transcribes audio using the FAL API.
    Handles audio upload, transcription job submission, and result formatting.
    """
    try:
        logging.info("🎤 Starting FAL transcription...")
        if not os.getenv("FAL_KEY"):
            return "[FAL_KEY not configured]"

        # Log audio details for debugging
        audio_size_mb = len(audio_bytes) / 1024 / 1024
        logging.info(f"📊 Audio size for transcription: {audio_size_mb:.2f}MB")

        if audio_size_mb < 0.001:
            logging.warning("⚠️ Warning: Audio file is extremely small, transcription quality may be poor")

        logging.info("📤 Uploading audio to FAL...")
        # FAL is robust; a generic 'audio/mpeg' is sufficient for MP3.
        url = fal_client.upload(data=audio_bytes, content_type="audio/mpeg")
        logging.info("✅ Upload successful to FAL")

        logging.info("🔄 Starting transcription job...")

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log_entry in update.logs:
                    logging.info(f"FAL: {log_entry['message']}")

        result = fal_client.subscribe(
            "fal-ai/whisper",
            arguments={
                "audio_url": url,
                "task": "transcribe",
                "language": None,
                "model": "base",
                "word_timestamps": False,
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Log transcription result details
        if hasattr(result, "get"):
            transcript_text = whisper_result_to_txt(result)
            logging.info(f"📝 Transcription result: {len(transcript_text)} characters")
            if len(transcript_text) < 100:
                logging.warning("⚠️ Warning: Transcription result is very short, audio quality may be poor")
        else:
            transcript_text = str(result)
            logging.info(f"📝 Raw transcription result type: {type(result)}")

        logging.info("✅ Transcription completed")
        return whisper_result_to_txt(result)

    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ FAL transcription failed: {error_msg}")
        if "403" in error_msg or "forbidden" in error_msg.lower():
            return "[FAL API access denied (403). Check API key permissions.]"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "[FAL API quota exceeded]"
        elif "timeout" in error_msg.lower():
            return "[FAL API timeout - audio may be too long]"
        else:
            return f"[FAL transcription failed: {error_msg}]"
