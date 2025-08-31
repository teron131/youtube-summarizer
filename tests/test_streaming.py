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

        resp = client.post(
            "/api/summarize-stream",
            json={"content": result_with_chapters.transcript_only_text[:2000], "content_type": "transcript"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.content.decode()
        assert "data:" in body


class TestSSEContract:
    """Unit-level SSE format checks (no external calls)."""

    @patch("app.stream_summarize_video")
    def test_sse_basic_contract(self, mock_stream, client):
        from youtube_summarizer.summarizer import Analysis, WorkflowState

        analysis = Analysis(title="SSE Test", summary="Test", takeaways=["t"], key_facts=["f"], chapters=[], keywords=["k"])
        chunk = WorkflowState(transcript_or_url="valid", analysis=analysis, iteration_count=1, is_complete=True)
        mock_stream.return_value = [chunk]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "OPENROUTER_API_KEY": "test"}):
            resp = client.post(
                "/api/summarize-stream",
                json={"content": "Valid content for SSE test", "content_type": "transcript"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.content.decode()
        assert "data:" in body
