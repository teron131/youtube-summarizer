"""
Comprehensive API Tests for YouTube Summarizer (Integration-first)
=================================================================

Uses real sample data from example_results.py instead of mocks.
Tests are gated by environment variables for external API usage.
"""

import os

import pytest


@pytest.mark.integration
class TestHealthAndInfo:
    """Basic health and info endpoints."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "YouTube Summarizer API"
        assert "version" in data
        assert "endpoints" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data


@pytest.mark.integration
class TestVideoAndScrap:
    """Video info and scraping tests using example_results.py."""

    def test_scrap_success(self, client):
        if not os.getenv("APIFY_API_KEY"):
            pytest.skip("APIFY_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        resp = client.post("/scrap", json={"url": result_with_chapters.url})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["url"].startswith("https://www.youtube.com/watch?v=")
        assert "transcript" in data
        assert "title" in data
        assert "author" in data
        # Test that response contains expected data types
        assert isinstance(data["title"], str)
        assert isinstance(data["author"], str)
        assert isinstance(data["transcript"], str)


@pytest.mark.integration
class TestSummarize:
    """Summarize endpoints using the LangGraph workflow (requires GEMINI_API_KEY or OPENROUTER_API_KEY)."""

    def test_summarize_success(self, client):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("GEMINI_API_KEY or OPENROUTER_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.transcript_only_text[:5000], "content_type": "transcript"}
        resp = client.post("/summarize", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "analysis" in data and "quality" in data
        assert "iteration_count" in data

    def test_summarize_stream_success(self, client):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("GEMINI_API_KEY or OPENROUTER_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        resp = client.post(
            "/stream-summarize",
            json={"content": result_with_chapters.transcript_only_text[:2000], "content_type": "transcript"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        content = resp.content.decode()
        assert "data:" in content


@pytest.mark.integration
class TestTwoStepWorkflow:
    """End-to-end workflow test (requires APIFY_API_KEY and GEMINI_API_KEY or OPENROUTER_API_KEY)."""

    def test_two_step(self, client):
        if not (os.getenv("APIFY_API_KEY") and (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))):
            pytest.skip("Required API keys not set; skipping integration test")

        from example_results import result_with_chapters

        # Step 1: scrape
        s1 = client.post("/scrap", json={"url": result_with_chapters.url})
        assert s1.status_code == 200
        d1 = s1.json()

        # Step 2: summarize using transcript
        s2 = client.post("/summarize", json={"content": d1["transcript"], "content_type": "transcript"})
        assert s2.status_code == 200
        d2 = s2.json()
        assert d1["status"] == "success"
        assert d2["status"] == "success"
        assert "analysis" in d2 and "quality" in d2
        assert isinstance(d2["analysis"], dict)
        assert isinstance(d2["quality"], dict)


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_url_scrap(self, client):
        """Test scraping with invalid URL."""
        response = client.post("/scrap", json={"url": "not-a-youtube-url"})
        assert response.status_code == 400
        data = response.json()
        assert data["detail"] == "Invalid YouTube URL"

    def test_empty_url_scrap(self, client):
        """Test scraping with empty URL."""
        response = client.post("/scrap", json={"url": ""})
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data or "errors" in data

    def test_missing_api_key_scrap(self, client):
        """Test scraping without API key."""
        from unittest.mock import patch

        with patch.dict("os.environ", {"APIFY_API_KEY": ""}):
            response = client.post("/scrap", json={"url": "https://youtube.com/watch?v=test"})
            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "Required API key missing"

    def test_missing_api_key_summarize(self, client):
        """Test summarization without API key."""
        from unittest.mock import patch

        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            response = client.post("/summarize", json={"content": "test content", "content_type": "transcript"})
            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "Required API key missing"

    def test_invalid_content_type(self, client):
        """Test summarization with invalid content type."""
        response = client.post("/summarize", json={"content": "test", "content_type": "invalid"})
        assert response.status_code == 422  # Pydantic validation error

    def test_content_too_short(self, client):
        """Test summarization with content too short."""
        response = client.post("/summarize", json={"content": "hi", "content_type": "transcript"})
        assert response.status_code == 422  # Pydantic validation error

    def test_content_too_long(self, client):
        """Test summarization with content too long."""
        long_content = "word " * 15000  # Exceeds 50k limit
        response = client.post("/summarize", json={"content": long_content, "content_type": "transcript"})
        assert response.status_code == 422  # Pydantic validation error
