"""
Streaming Tests for YouTube Summarizer (Integration-first)
=========================================================

Uses real sample data from example_results.py when API keys are present.
"""

import os
from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestLangGraphStreaming:
    """LangGraph streaming integration tests (requires GEMINI_API_KEY or OPENROUTER_API_KEY)."""

    def test_stream_integration(self, client):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("GEMINI_API_KEY or OPENROUTER_API_KEY not set; skipping integration test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.transcript_only_text[:2000], "content_type": "transcript", "target_language": "en", "analysis_model": "google/gemini-2.5-pro", "quality_model": "google/gemini-2.5-flash"}
        resp = client.post("/stream-summarize", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.content.decode()
        assert "data:" in body


class TestSSEContract:
    """Unit-level SSE format checks (no external calls)."""

    @patch("app.stream_summarize_video")
    def test_sse_basic_contract(self, mock_stream, client):
        from youtube_summarizer.summarizer import (
            Analysis,
            Chapter,
            GraphState,
            Quality,
            Rate,
        )

        # Create Rate objects for quality assessment
        test_rate = Rate(rate="Pass", reason="Test reason")

        # Create Quality object
        quality = Quality(completeness=test_rate, structure=test_rate, grammar=test_rate, no_garbage=test_rate, useful_keywords=test_rate, correct_language=test_rate)

        # Create analysis with proper structure
        analysis = Analysis(title="SSE Test", summary="Test summary for streaming analysis", takeaways=["Test takeaway 1", "Test takeaway 2"], key_facts=["Test fact 1", "Test fact 2"], chapters=[], keywords=["test", "sse"], target_language="en")

        chunk = GraphState(
            transcript_or_url="valid",
            chapters=[],
            analysis=analysis,
            quality=quality,
            iteration_count=1,
            target_language="en",
            analysis_model="google/gemini-2.5-pro",
            quality_model="google/gemini-2.5-flash",
            is_complete=True,
        )
        mock_stream.return_value = [chunk]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "OPENROUTER_API_KEY": "test"}):
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": "Valid content for SSE test",
                    "content_type": "transcript",
                    "target_language": "en",
                    "analysis_model": "google/gemini-2.5-pro",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.content.decode()
        assert "data:" in body
