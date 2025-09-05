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
def mock_quality_result():
    """Mock quality assessment result."""
    from youtube_summarizer.summarizer import Quality, Rate

    return Quality(completeness=Rate(rate="Pass", reason="Complete analysis"), structure=Rate(rate="Pass", reason="Well structured"), grammar=Rate(rate="Pass", reason="Good grammar"), no_garbage=Rate(rate="Pass", reason="Clean content"), meta_language_avoidance=Rate(rate="Pass", reason="No meta language"), useful_keywords=Rate(rate="Pass", reason="Useful keywords"), correct_language=Rate(rate="Pass", reason="Correct language"))


@pytest.fixture
def mock_graph_state(mock_analysis_result, mock_quality_result):
    """Mock GraphState for testing."""
    from youtube_summarizer.summarizer import GraphState

    return GraphState(transcript_or_url="Sample transcript for testing", chapters=[{"title": "Introduction", "timeDescription": "0:00"}, {"title": "Main Content", "timeDescription": "2:30"}], analysis_model="google/gemini-2.5-pro", quality_model="google/gemini-2.5-flash", target_language="en", analysis=mock_analysis_result, quality=mock_quality_result, iteration_count=1, is_complete=True)


@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_llm
    return mock_llm


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return """
    Welcome to this video about artificial intelligence and machine learning.
    Today we're going to explore the fundamentals of AI technology and its applications.

    Artificial intelligence refers to the simulation of human intelligence in machines.
    Machine learning is a subset of AI that enables systems to learn from data.

    There are several types of machine learning including supervised, unsupervised, and reinforcement learning.
    Deep learning uses neural networks with multiple layers to process complex data.

    AI has applications in healthcare, finance, transportation, and many other industries.
    The future of AI looks promising with continued advancements in the field.
    """


@pytest.fixture
def sample_chapters():
    """Sample video chapters for testing."""
    return [{"title": "Introduction to AI", "timeDescription": "0:00"}, {"title": "Machine Learning Basics", "timeDescription": "2:15"}, {"title": "Deep Learning", "timeDescription": "5:30"}, {"title": "AI Applications", "timeDescription": "8:45"}, {"title": "Future of AI", "timeDescription": "11:20"}]


@pytest.fixture
def clean_env():
    """Fixture to clear environment variables for testing."""
    original_env = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)
