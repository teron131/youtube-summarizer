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
    from youtube_summarizer.summarizer import Analysis, Chapter

    # Create real Pydantic objects that match the current structure
    return Analysis(title="Test Video Analysis", summary="This is a comprehensive analysis of the video content covering key concepts and practical applications.", takeaways=["First key takeaway about the main topic", "Second important insight from the content", "Third actionable point for implementation"], key_facts=["Important statistic or data point from the video", "Key fact that supports the main argument", "Essential piece of information for understanding"], chapters=[Chapter(header="Introduction to Main Topic", summary="Overview of the primary subject matter and its importance in the field.", key_points=["Point 1: Basic foundation concepts", "Point 2: Key principles and methodologies", "Point 3: Practical applications and examples"]), Chapter(header="Advanced Implementation", summary="Detailed exploration of advanced techniques and best practices.", key_points=["Point 1: Advanced implementation strategies", "Point 2: Common challenges and solutions", "Point 3: Optimization techniques"])], keywords=["test", "analysis", "video", "summary"], target_language=None)


@pytest.fixture
def clean_env():
    """Fixture to clear environment variables for testing."""
    original_env = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
