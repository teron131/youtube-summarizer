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
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data


@pytest.mark.integration
class TestVideoAndScrap:
    """Video info and scraping tests using example_results.py."""

    def test_video_info_success(self, client):
        if not os.getenv("APIFY_API_KEY"):
            pytest.skip("APIFY_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        resp = client.post("/api/video-info", json={"url": result_with_chapters.url})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["title"], str)
        assert isinstance(data["author"], str)

    def test_scrap_success(self, client):
        if not os.getenv("APIFY_API_KEY"):
            pytest.skip("APIFY_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        resp = client.post("/api/scrap", json={"url": result_with_chapters.url})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["cleaned_url"].startswith("https://www.youtube.com/watch?v=")
        assert "video_info" in data
        assert "transcript" in data


@pytest.mark.integration
class TestSummarize:
    """Summarize endpoints using the LangGraph workflow (requires GEMINI_API_KEY or OPENROUTER_API_KEY)."""

    def test_summarize_success(self, client):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("GEMINI_API_KEY or OPENROUTER_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.transcript_only_text[:5000], "content_type": "transcript"}
        resp = client.post("/api/summarize", json=payload)
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
            "/api/summarize-stream",
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
        s1 = client.post("/api/scrap", json={"url": result_with_chapters.url})
        assert s1.status_code == 200
        d1 = s1.json()

        # Step 2: summarize using transcript
        s2 = client.post("/api/summarize", json={"content": d1["transcript"], "content_type": "transcript"})
        assert s2.status_code == 200
        d2 = s2.json()
        assert d1["status"] == "success"
        assert d2["status"] == "success"
        assert "analysis" in d2 and "quality" in d2
