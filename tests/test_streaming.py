"""
Streaming Tests for YouTube Summarizer (Integration-first)
=========================================================

Uses real sample data from example_results.py when API keys are present.
Updated for app.py v3.0.0 streaming enhancements and progress tracking.
"""

import json
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

        # Validate streaming response structure
        lines = body.split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) > 0

        # Check for completion marker
        completion_found = any("complete" in line.lower() for line in data_lines)
        assert completion_found, "Stream should contain completion marker"

    def test_stream_with_chapters_extraction(self, client):
        """Test streaming with YouTube URL that includes chapters."""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("API keys not set; skipping chapters test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.url, "content_type": "url", "target_language": "en", "analysis_model": "google/gemini-2.5-flash", "quality_model": "google/gemini-2.5-flash"}
        resp = client.post("/stream-summarize", json=payload)
        assert resp.status_code == 200
        body = resp.content.decode()

        # Should contain chapter-related data
        assert "data:" in body
        data_lines = [line for line in body.split("\n") if line.startswith("data: ")]

        # At least some data chunks should be present
        assert len(data_lines) >= 2

    def test_stream_progress_tracking(self, client):
        """Test that streaming includes proper progress tracking."""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("API keys not set; skipping progress test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.transcript_only_text[:1500], "content_type": "transcript", "target_language": "en", "analysis_model": "google/gemini-2.5-flash", "quality_model": "google/gemini-2.5-flash"}
        resp = client.post("/stream-summarize", json=payload)
        assert resp.status_code == 200
        body = resp.content.decode()

        # Parse streaming data
        data_lines = [line for line in body.split("\n") if line.startswith("data: ")]
        json_chunks = []

        for line in data_lines:
            try:
                chunk_data = line.replace("data: ", "")
                if chunk_data.strip():
                    parsed = json.loads(chunk_data)
                    json_chunks.append(parsed)
            except json.JSONDecodeError:
                continue

        # Should have multiple progress updates
        assert len(json_chunks) >= 3

        # Check for progress tracking fields
        has_iteration_count = any("iteration_count" in chunk for chunk in json_chunks)
        has_timestamp = any("timestamp" in chunk for chunk in json_chunks)

        assert has_iteration_count, "Stream should include iteration count tracking"
        assert has_timestamp, "Stream should include timestamps"

    def test_stream_model_metadata(self, client):
        """Test that streaming includes model metadata."""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
            pytest.skip("API keys not set; skipping metadata test")

        from example_results import result_with_chapters

        payload = {"content": result_with_chapters.transcript_only_text[:1000], "content_type": "transcript", "target_language": "zh-TW", "analysis_model": "google/gemini-2.5-pro", "quality_model": "google/gemini-2.5-flash"}
        resp = client.post("/stream-summarize", json=payload)
        assert resp.status_code == 200
        body = resp.content.decode()

        # Parse final chunk to check metadata
        lines = body.split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        # Look for completion chunk
        completion_chunks = []
        for line in data_lines:
            try:
                chunk_data = line.replace("data: ", "")
                if chunk_data.strip():
                    parsed = json.loads(chunk_data)
                    if parsed.get("type") == "complete" or parsed.get("is_complete"):
                        completion_chunks.append(parsed)
            except json.JSONDecodeError:
                continue

        if completion_chunks:
            final_chunk = completion_chunks[-1]
            # Should include processing time and chunk count
            assert "processing_time" in final_chunk or "total_chunks" in final_chunk


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

    @patch("app.stream_summarize_video")
    def test_sse_chunk_validation(self, mock_stream, client):
        """Test SSE chunk structure and JSON validation."""
        from youtube_summarizer.summarizer import GraphState

        # Create mock chunks with different states
        chunks = [
            GraphState(
                transcript_or_url="test content",
                analysis=None,
                quality=None,
                iteration_count=1,
                is_complete=False,
            ),
            GraphState(
                transcript_or_url="test content",
                analysis=None,
                quality=None,
                iteration_count=1,
                is_complete=True,
            ),
        ]
        mock_stream.return_value = chunks

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": "test",
                    "content_type": "transcript",
                    "analysis_model": "google/gemini-2.5-flash",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )

        assert resp.status_code == 200
        body = resp.content.decode()

        # Parse SSE data
        lines = body.split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        # Should have at least 2 chunks
        assert len(data_lines) >= 2

        # Validate JSON structure of each chunk
        for line in data_lines:
            chunk_data = line.replace("data: ", "")
            if chunk_data.strip():
                try:
                    parsed = json.loads(chunk_data)
                    assert "timestamp" in parsed
                    assert "chunk_number" in parsed
                except json.JSONDecodeError:
                    # Skip malformed chunks in this test
                    continue

    @patch("app.stream_summarize_video")
    def test_sse_large_content_truncation(self, mock_stream, client):
        """Test that large content is properly truncated in SSE responses."""
        from youtube_summarizer.summarizer import GraphState

        # Create large content
        large_content = "word " * 2000  # Large content that should be truncated

        chunk = GraphState(
            transcript_or_url=large_content,
            analysis=None,
            quality=None,
            iteration_count=1,
            is_complete=True,
        )
        mock_stream.return_value = [chunk]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": large_content,
                    "content_type": "transcript",
                    "analysis_model": "google/gemini-2.5-flash",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )

        assert resp.status_code == 200
        body = resp.content.decode()

        # Verify that the response is not excessively large (should be truncated)
        assert len(body) < len(large_content) * 2

        # Check for truncation indicators
        assert "truncated" in body.lower() or len(body) < len(large_content)


@pytest.mark.unit
class TestStreamingErrorHandling:
    """Test streaming error handling and edge cases."""

    @patch("app.stream_summarize_video")
    def test_stream_error_handling(self, mock_stream, client):
        """Test streaming error handling."""
        # Mock an exception in streaming
        mock_stream.side_effect = Exception("Test streaming error")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": "test content",
                    "content_type": "transcript",
                    "analysis_model": "google/gemini-2.5-flash",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )

        assert resp.status_code == 200  # Still returns 200 but with error data
        body = resp.content.decode()

        # Should contain error data
        assert "error" in body.lower() or "failed" in body.lower()

    def test_stream_without_api_keys(self, client):
        """Test streaming without required API keys."""
        resp = client.post(
            "/stream-summarize",
            json={
                "content": "test content",
                "content_type": "transcript",
                "analysis_model": "google/gemini-2.5-flash",
                "quality_model": "google/gemini-2.5-flash",
            },
        )

        assert resp.status_code == 500
        data = resp.json()
        assert "Required API key missing" in data["detail"]

    @patch("app.stream_summarize_video")
    def test_stream_with_malformed_data(self, mock_stream, client):
        """Test streaming with malformed response data."""
        from youtube_summarizer.summarizer import GraphState

        # Create a chunk that will cause JSON serialization issues
        chunk = GraphState(
            transcript_or_url="test",
            analysis=None,
            quality=None,
            iteration_count=1,
            is_complete=True,
        )

        # Mock the stream to return problematic data
        mock_stream.return_value = [chunk]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": "test content",
                    "content_type": "transcript",
                    "analysis_model": "google/gemini-2.5-flash",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )

        assert resp.status_code == 200
        body = resp.content.decode()

        # Should still produce valid SSE format even with issues
        assert "data:" in body

    def test_stream_cors_headers(self, client):
        """Test CORS headers on streaming endpoint."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            with patch("app.stream_summarize_video") as mock_stream:
                mock_stream.return_value = []

                resp = client.post(
                    "/stream-summarize",
                    json={
                        "content": "test",
                        "content_type": "transcript",
                        "analysis_model": "google/gemini-2.5-flash",
                        "quality_model": "google/gemini-2.5-flash",
                    },
                )

                # Check CORS headers
                assert resp.headers.get("access-control-allow-origin") == "*"
                assert resp.headers.get("x-accel-buffering") == "no"
                assert resp.headers.get("cache-control") == "no-cache"
                assert resp.headers.get("connection") == "keep-alive"


@pytest.mark.unit
class TestStreamingValidation:
    """Test streaming request validation."""

    def test_stream_content_validation(self, client):
        """Test content validation for streaming endpoint."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            # Test empty content
            resp = client.post("/stream-summarize", json={"content": "", "content_type": "transcript"})
            assert resp.status_code == 422

            # Test content too short
            resp = client.post("/stream-summarize", json={"content": "hi", "content_type": "transcript"})
            assert resp.status_code == 422

            # Test invalid content type
            resp = client.post("/stream-summarize", json={"content": "test content", "content_type": "invalid"})
            assert resp.status_code == 422

    def test_stream_model_validation(self, client):
        """Test model validation for streaming endpoint."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            # Test invalid model name
            resp = client.post(
                "/stream-summarize",
                json={
                    "content": "test content",
                    "content_type": "transcript",
                    "analysis_model": "invalid-model-name",
                    "quality_model": "google/gemini-2.5-flash",
                },
            )
            assert resp.status_code == 422
