"""
Test Configuration and Fixtures
===============================

Shared pytest fixtures and configuration for all tests.
"""

import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (may require API keys)")
    config.addinivalue_line("markers", "unit: marks tests as unit tests (no external dependencies)")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "skip: marks tests to be skipped")


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    # Import here to avoid module loading issues
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from app import app
        from fastapi.testclient import TestClient

        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"Could not import app dependencies: {e}")


@pytest.fixture
def mock_channel():
    """Mock Channel object for testing."""
    # Create a mock object that mimics the Channel structure
    mock = MagicMock()
    mock.id = "UCtest123"
    mock.url = "https://youtube.com/channel/UCtest123"
    mock.handle = "@testchannel"
    mock.title = "Rick Astley"
    return mock


@pytest.fixture
def mock_youtube_scrapper_result(mock_channel):
    """Mock YouTubeScrapperResult object for testing."""
    # Create a mock object that mimics the YouTubeScrapperResult structure
    mock = MagicMock()
    mock.id = "dQw4w9WgXcQ"
    mock.thumbnail = "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
    mock.url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
    mock.type = "video"
    mock.title = "Never Gonna Give You Up"
    mock.description = "Rick Astley's official music video for Never Gonna Give You Up"
    mock.commentCountInt = 150000
    mock.likeCountText = "15M"
    mock.likeCountInt = 15000000
    mock.viewCountText = "1.4B views"
    mock.viewCountInt = 1400000000
    mock.publishDateText = "2009-10-25"
    mock.publishDate = datetime(2009, 10, 25)
    mock.channel = mock_channel
    mock.chapters = []
    mock.watchNextVideos = []
    mock.keywords = ["music", "rick astley", "never gonna give you up"]
    mock.durationMs = 213000
    mock.durationFormatted = "3:33"
    mock.transcript = []
    mock.transcript_only_text = "Never gonna give you up, never gonna let you down..."
    mock.parsed_transcript = "Never gonna give you up, never gonna let you down..."
    mock.language = "en"
    return mock


@pytest.fixture
def mock_analysis_result():
    """Mock analysis result for Gemini API responses."""
    from youtube_summarizer.summarizer import Chapter, TimestampedText

    # Create mock objects that mimic the new structure
    mock = MagicMock()
    mock.title = "Test Video Analysis"
    mock.summary = "This is a comprehensive analysis of the video content."

    # Create TimestampedText objects for key_facts and takeaways
    mock.key_facts = [TimestampedText(text="Fact 1", timestamp="00:01:23"), TimestampedText(text="Fact 2", timestamp="00:02:45")]
    mock.takeaways = [TimestampedText(text="Takeaway 1", timestamp="00:03:12"), TimestampedText(text="Takeaway 2", timestamp="00:04:56")]

    # Mock chapter with optional timestamp
    mock_chapter = MagicMock()
    mock_chapter.header = "Main Content"
    mock_chapter.key_points = ["Point 1", "Point 2", "Point 3"]
    mock_chapter.summary = "This is a comprehensive analysis of the video content."
    mock_chapter.timestamp = "00:00:30"

    mock.chapters = [mock_chapter]
    return mock


@pytest.fixture
def clean_env():
    """Fixture to clear environment variables for testing."""
    original_env = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
