"""
Audio Transcription Service
---------------------------

This module is responsible for handling audio transcription. It includes functions
for optimizing audio files for transcription and for interfacing with the FAL
transcription service.
"""

import io
import os

import fal_client
from pydub import AudioSegment

from .utils import log_and_print, whisper_result_to_txt


def optimize_audio_for_transcription(audio_bytes: bytes) -> bytes:
    """
    Optimizes audio for transcription using a reliable, high-compression setting.
    Converts audio to mono, sets a 16kHz sample rate, and uses a 32kbps bitrate
    to ensure a small file size suitable for transcription APIs.
    """
    raw_size_mb = len(audio_bytes) / 1024 / 1024
    log_and_print(f"🔄 Optimizing audio ({raw_size_mb:.1f}MB) for transcription...")

    try:
        audio_io = io.BytesIO(audio_bytes)
        audio_segment = None
        # Try loading with common formats first
        for fmt in ["mp3", "mp4", "m4a", "webm", "ogg"]:
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io, format=fmt)
                log_and_print(f"✅ Loaded audio source as {fmt}")
                break
            except Exception:
                continue

        # Fallback to auto-detection if specific formats fail
        if not audio_segment:
            audio_io.seek(0)
            audio_segment = AudioSegment.from_file(audio_io)
            log_and_print("✅ Loaded audio source with auto-detection")

        # Apply a standard, effective compression configuration
        log_and_print("⚙️ Applying standard compression: 32kbps, mono, 16kHz sample rate")
        compressed_audio = audio_segment.set_channels(1).set_frame_rate(16000)

        output_buffer = io.BytesIO()
        compressed_audio.export(output_buffer, format="mp3", bitrate="32k")
        compressed_bytes = output_buffer.getvalue()

        compressed_size_mb = len(compressed_bytes) / 1024 / 1024
        log_and_print(f"✅ Compression complete. New size: {compressed_size_mb:.1f}MB")

        return compressed_bytes

    except Exception as e:
        log_and_print(f"❌ Audio optimization failed: {e}")
        log_and_print(f"⚠️ Could not compress audio. Using original audio file ({raw_size_mb:.1f}MB).")
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
            arguments={"audio_url": url, "task": "transcribe", "language": None},
            with_logs=True,
            on_queue_update=on_queue_update,
        )

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
