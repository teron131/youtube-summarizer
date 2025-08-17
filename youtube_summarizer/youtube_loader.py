import io
import time

from dotenv import load_dotenv
from pydub import AudioSegment
from pytubefix import Buffer, YouTube

from .transcriber import transcribe_with_fal
from .utils import whisper_result_to_txt

load_dotenv()


def get_best_audio_stream(youtube: YouTube, attempt: int = 0):
    """Get the best available audio stream with multiple fallback strategies."""
    strategies = [
        # Strategy 1: Get highest quality audio-only stream
        lambda: youtube.streams.get_audio_only(),
        # Strategy 2: Filter audio-only and get highest bitrate (with validation)
        lambda: youtube.streams.filter(only_audio=True).filter(lambda s: s.abr is not None).order_by("abr").desc().first(),
        # Strategy 3: Get any audio-only stream
        lambda: youtube.streams.filter(only_audio=True).first(),
        # Strategy 4: Get lowest bitrate audio (most reliable)
        lambda: youtube.streams.filter(only_audio=True).filter(lambda s: s.abr is not None).order_by("abr").asc().first(),
        # Strategy 5: Get any stream with audio (including video+audio)
        lambda: youtube.streams.filter(adaptive=False).filter(lambda s: s.includes_audio_track).first(),
    ]

    # Try strategies in order, cycling through them on retries
    strategy_index = attempt % len(strategies)

    for i in range(len(strategies)):
        current_strategy = (strategy_index + i) % len(strategies)
        try:
            print(f"Trying stream strategy {current_strategy + 1}")
            stream = strategies[current_strategy]()

            if stream:
                print(f"Found stream: {stream}")
                print(f"  - Codec: {getattr(stream, 'codecs', 'Unknown')}")
                print(f"  - Bitrate: {getattr(stream, 'abr', 'Unknown')}")
                print(f"  - File size: {getattr(stream, 'filesize', 'Unknown')}")
                return stream

        except Exception as e:
            print(f"Strategy {current_strategy + 1} failed: {e}")
            continue

    return None


def youtube_to_audio_bytes(youtube: YouTube, max_retries: int = 5) -> bytes:
    """Get audio bytes from YouTube object with enhanced retry logic.
    Args:
        youtube (YouTube): YouTube object
        max_retries (int): Maximum number of retry attempts
    Returns:
        bytes: Audio bytes
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\n--- Audio Download Attempt {attempt + 1}/{max_retries} ---")

            # Get audio stream with intelligent fallback
            youtube_stream = get_best_audio_stream(youtube, attempt)

            if not youtube_stream:
                raise RuntimeError("No audio stream available after trying all strategies")

            print(f"Selected stream: {youtube_stream}")

            # Validate stream properties before download
            if hasattr(youtube_stream, "filesize") and youtube_stream.filesize:
                print(f"Expected download size: {youtube_stream.filesize / 1024 / 1024:.1f} MB")

            # Download with progress indication and error isolation
            print("Downloading audio stream...")
            try:
                buffer = Buffer()

                # Add pre-download validation to catch NoneType issues early
                if hasattr(youtube_stream, "url") and youtube_stream.url:
                    print(f"Stream URL available: {youtube_stream.url[:50]}...")
                else:
                    raise RuntimeError("Stream URL is None or invalid")

                # Check for problematic None values that cause int() errors
                stream_props = ["filesize", "fps", "resolution", "abr"]
                for prop in stream_props:
                    if hasattr(youtube_stream, prop):
                        value = getattr(youtube_stream, prop)
                        if value is None and prop in ["filesize"]:  # filesize being None causes issues
                            print(f"⚠️ Warning: {prop} is None, may cause download issues")

                buffer.download_in_buffer(youtube_stream)
                audio_data = buffer.read()

            except Exception as download_e:
                # More specific error handling for the download step
                error_msg = str(download_e)
                if "int() argument must be a string" in error_msg and "NoneType" in error_msg:
                    raise RuntimeError(f"Download failed due to None value in stream properties. This often happens with token-based streams. Original error: {error_msg}")
                else:
                    raise RuntimeError(f"Download failed: {error_msg}")

            if not audio_data:
                raise RuntimeError("Downloaded audio data is empty")

            print(f"Downloaded {len(audio_data)} bytes of audio data")

            # Process audio with error handling
            print("Processing audio with AudioSegment...")
            try:
                with io.BytesIO(audio_data) as in_memory_file:
                    # Try different format hints for AudioSegment
                    audio_segment = None
                    format_hints = ["mp4", "webm", "m4a", None]  # None = auto-detect

                    for fmt in format_hints:
                        try:
                            if fmt:
                                audio_segment = AudioSegment.from_file(in_memory_file, format=fmt)
                            else:
                                audio_segment = AudioSegment.from_file(in_memory_file)
                            print(f"Successfully processed audio with format hint: {fmt or 'auto-detect'}")
                            break
                        except Exception as fmt_e:
                            print(f"Format {fmt or 'auto-detect'} failed: {fmt_e}")
                            in_memory_file.seek(0)  # Reset for next attempt
                            continue

                    if not audio_segment:
                        raise RuntimeError("Could not process audio with any format")

            except Exception as audio_e:
                raise RuntimeError(f"AudioSegment processing failed: {audio_e}")

            # Export to MP3 with validation
            print("Exporting to MP3...")
            try:
                with io.BytesIO() as output_buffer:
                    # Use more conservative export settings
                    export_kwargs = {"format": "mp3", "bitrate": "32k", "parameters": ["-ac", "1"]}  # Lower bitrate for reliability  # Force mono to reduce size

                    audio_segment.export(output_buffer, **export_kwargs)
                    result = output_buffer.getvalue()

                    if not result:
                        raise RuntimeError("Export produced empty result")

                    print(f"Successfully exported {len(result)} bytes of MP3 data")
                    return result

            except Exception as export_e:
                raise RuntimeError(f"MP3 export failed: {export_e}")

        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                # Progressive wait time with some randomization
                base_wait = min(2**attempt, 10)  # Cap at 10 seconds
                wait_time = base_wait + (attempt * 2)  # Add progressive component
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

                # Force refresh the YouTube object streams on some attempts
                if attempt % 2 == 1:
                    try:
                        print("Refreshing stream information...")
                        youtube.check_availability()
                    except:
                        pass  # Ignore refresh errors

    raise RuntimeError(f"Failed to download audio after {max_retries} attempts. Last error: {last_exception}")


def youtube_to_subtitle(youtube: YouTube, language: str = None) -> str:
    """Process a YouTube video: try to get captions first, fall back to audio transcription."""

    # Check for available captions first
    print("🔍 Checking for available captions...")
    available_captions = list(youtube.captions.keys())
    print(f"📝 Available captions: {available_captions}")

    # Priority order for captions (English first, then others)
    caption_priorities = ["zh-HK", "zh-CN", "en", "a.en"]

    for caption_key in caption_priorities:
        if caption_key in available_captions:
            try:
                print(f"✅ Using caption: {caption_key}")
                caption = youtube.captions[caption_key]
                caption_text = caption.generate_txt_captions()

                if caption_text and caption_text.strip():
                    print(f"📖 Caption length: {len(caption_text)} characters")
                    return caption_text
                else:
                    print(f"⚠️ Caption {caption_key} is empty, trying next option...")
            except Exception as e:
                print(f"❌ Failed to get caption {caption_key}: {e}")
                continue

    # If no captions available or all failed, fall back to transcription
    print("🎵 No captions available, falling back to audio transcription...")
    audio_bytes = youtube_to_audio_bytes(youtube)
    result = transcribe_with_fal(audio_bytes, language)
    subtitle = whisper_result_to_txt(result)
    return subtitle


# Main function
def youtube_loader(url: str, max_retries: int = 3) -> str:
    """Load and process a YouTube video's subtitle, title, and author information from a URL.
    Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url (str): The YouTube video URL to load
        max_retries (int): Maximum number of retry attempts for the entire process

    Returns:
        str: Formatted string containing the video title, author and subtitle
    """
    print(f"\n🎬 Loading YouTube video: {url}")
    print("=" * 50)

    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\n📡 MAIN ATTEMPT {attempt + 1}/{max_retries}")

            # Try with tokens first, then without if that fails
            token_strategies = [True, False] if attempt == 0 else [False]

            for use_tokens in token_strategies:
                try:
                    strategy_name = "with tokens" if use_tokens else "without tokens"
                    print(f"🔑 Trying {strategy_name}...")

                    youtube = YouTube(url)

                    # Get video metadata first to check if the video is accessible
                    print("📋 Fetching video metadata...")
                    title = youtube.title
                    author = youtube.author
                    duration = getattr(youtube, "length", "Unknown")

                    print(f"✅ Video accessible:")
                    print(f"   📺 Title: {title}")
                    print(f"   👤 Author: {author}")
                    print(f"   ⏱️  Duration: {duration}s")

                    # Show available streams for debugging and detect problematic streams
                    try:
                        all_streams = youtube.streams.filter(only_audio=True)
                        print(f"🎵 Found {len(all_streams)} audio streams")

                        # Check for problematic stream conditions that cause NoneType errors
                        problematic_streams = 0
                        for i, stream in enumerate(all_streams[:3]):  # Show first 3
                            print(f"   Stream {i+1}: {stream}")

                            # Check for conditions that often cause NoneType errors with tokens
                            if hasattr(stream, "sabr") and stream.sabr == True:
                                problematic_streams += 1

                        # If most streams are problematic and we're using tokens, suggest fallback
                        if use_tokens and problematic_streams >= 2:
                            print("⚠️  Detected many streams with 'sabr=True' - these often cause NoneType errors with tokens")
                            print("💡 Recommendation: Skip to no-token approach for faster success")
                            raise RuntimeError("Preemptive fallback due to problematic stream conditions")

                    except Exception as stream_e:
                        if "Preemptive fallback" in str(stream_e):
                            raise stream_e  # Re-raise our intentional fallback
                        print("⚠️  Could not enumerate streams")

                    # Now try to get the subtitle with enhanced audio processing
                    print("\n🎯 Starting audio processing...")
                    formatted_subtitle = youtube_to_subtitle(youtube, language=None)

                    content = [
                        "Answer the user's question based on the full content.",
                        f"Title: {title}",
                        f"Author: {author}",
                        f"subtitle:\n{formatted_subtitle}",
                    ]

                    print("✅ SUCCESS: Video processed successfully!")
                    return "\n".join(content)

                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Strategy failed ({strategy_name}): {error_msg}")

                    # Don't immediately fail if using tokens - try without tokens
                    if use_tokens:
                        print("🔄 Falling back to no-token approach...")
                        continue
                    else:
                        # Both strategies failed for this attempt
                        raise e

        except Exception as e:
            last_exception = e
            print(f"💥 FULL ATTEMPT {attempt + 1} FAILED: {e}")

            if attempt < max_retries - 1:
                # Intelligent wait time with backoff
                base_wait = 3 + (attempt * 2)  # 3, 5, 7 seconds
                wait_time = base_wait + min(attempt * 3, 10)  # Add progressive component, cap at 10
                print(f"⏳ Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)

                # Add some buffer time for YouTube's rate limiting
                if attempt > 0:
                    print("🛡️  Adding buffer time for rate limiting...")
                    time.sleep(2)

    raise RuntimeError(f"Failed to load YouTube video after {max_retries} attempts. Last error: {last_exception}")
