"""
Summarization route handlers
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from models.requests import SummarizeRequest, SummarizeResponse
from youtube_summarizer.summarizer import (
    SummarizerOutput,
    SummarizerState,
    create_graph,
    stream_summarize_video,
)
from youtube_summarizer.utils import is_youtube_url, serialize_nested
from youtube_summarizer.youtube_scrapper import scrap_youtube

from .helpers import get_processing_time, parse_scraper_result, run_async_task


async def summarize_handler(request: SummarizeRequest) -> SummarizeResponse:
    """Generate AI analysis using LangGraph workflow."""
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Required API key missing")

    start_time = datetime.now()

    try:
        # Handle content validation
        if request.content_type == "url":
            if not request.content or not is_youtube_url(request.content):
                raise HTTPException(status_code=400, detail="Valid YouTube URL required")
            validated_content = request.content
        else:
            if not request.content or not request.content.strip():
                raise HTTPException(status_code=400, detail="Content required")
            validated_content = request.content

        # Prepare initial state
        if request.content_type == "url":
            scrap_result = await run_async_task(scrap_youtube, validated_content)
            parsed_data = parse_scraper_result(scrap_result)
            transcript = parsed_data.get("transcript") or validated_content
        else:
            transcript = validated_content

        initial_state = SummarizerState(
            transcript=transcript,
            target_language=request.target_language,
        )

        # Invoke the graph
        graph = create_graph()
        result_dict = await run_async_task(graph.invoke, initial_state.model_dump())
        output = SummarizerOutput.model_validate(result_dict)

        return SummarizeResponse(
            status="success",
            message="Analysis completed successfully",
            analysis=output.analysis,
            quality=output.quality,
            processing_time=get_processing_time(start_time),
            iteration_count=output.iteration_count,
            target_language=request.target_language or "en",
            analysis_model=request.analysis_model,
            quality_model=request.quality_model,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:100]}")


async def stream_summarize_handler(request: SummarizeRequest) -> StreamingResponse:
    """Stream analysis with real-time progress updates."""
    if not os.getenv("GEMINI_API_KEY"):
        # Using HTTPException here instead of create_error_response which wasn't defined
        raise HTTPException(status_code=500, detail="Config missing: GEMINI_API_KEY")

    async def generate_stream():
        start_time = datetime.now()

        try:
            # Handle content validation based on content_type
            if request.content_type == "url":
                # For URL type, we need to scrape to get transcript
                if not request.content or not is_youtube_url(request.content):
                    raise HTTPException(status_code=400, detail="Valid YouTube URL is required when content_type is 'url'")

                # Scrape the video to get transcript
                logging.info(f"🔗 Scraping YouTube video: {request.content}")
                scrap_result = await run_async_task(scrap_youtube, request.content)
                parsed_data = parse_scraper_result(scrap_result)

                # Use transcript if available, otherwise send the YouTube URL
                if parsed_data.get("transcript"):
                    validated_content = parsed_data["transcript"]
                    logging.info("📝 Using scraped transcript for analysis")
                else:
                    # No transcript available - send YouTube URL to Gemini
                    validated_content = request.content
                    logging.info("🎯 No transcript available - sending YouTube URL to Gemini")

                    # Force Gemini models when no transcript is available
                    if not request.analysis_model.startswith("google/"):
                        logging.info("🔄 Switching to Gemini for analysis (no transcript available)")
                        request.analysis_model = "google/gemini-2.5-pro"
                    if not request.quality_model.startswith("google/"):
                        logging.info("🔄 Switching to Gemini for quality check (no transcript available)")
                        request.quality_model = "google/gemini-2.5-flash"

            else:
                # For transcript type, content is required
                if not request.content or request.content.strip() == "":
                    raise HTTPException(status_code=400, detail="Content is required when content_type is 'transcript'")
                validated_content = request.content

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for client processing

            # Use stream_summarize_video for streaming
            # Prepare content for streaming
            if request.content_type == "url":
                # Extract transcript if needed - wait, we already did this above?
                # Ah, the logic in app.py was duplicated. We already have 'validated_content' from above.
                # If we scraped above, validated_content IS the transcript or URL.
                stream_content = validated_content
            else:
                stream_content = validated_content

            chunk_count = 0
            final_state = None

            # Stream using stream_summarize_video
            for state_chunk in stream_summarize_video(stream_content, request.target_language):
                try:
                    # Convert chunk to dict efficiently
                    chunk_dict = state_chunk.model_dump() if hasattr(state_chunk, "model_dump") else {}

                    # Store final state reference (avoid copying large objects)
                    final_state = chunk_dict

                    # Serialize and stream chunk in one operation
                    serialized_chunk = serialize_nested(chunk_dict)
                    serialized_chunk["timestamp"] = datetime.now().isoformat()
                    serialized_chunk["chunk_number"] = chunk_count

                    yield f"data: {json.dumps(serialized_chunk, ensure_ascii=False)}\n\n"
                    chunk_count += 1

                    # Minimal adaptive delay for client processing
                    if chunk_count % 10 == 0:  # Only delay every 10 chunks
                        await asyncio.sleep(0.01)

                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    logging.warning(f"⚠️ Failed to serialize chunk {chunk_count}: {str(e)}")
                    chunk_count += 1
                    continue

            # Send completion message with final analysis and quality data
            completion_data = {
                "type": "complete",
                "message": "Analysis completed",
                "processing_time": get_processing_time(start_time),
                "total_chunks": chunk_count,
                "timestamp": datetime.now().isoformat(),
            }

            # Safely extract final state data if available
            if final_state and isinstance(final_state, dict):
                for key in ["analysis", "quality", "iteration_count", "is_complete"]:
                    if key in final_state and final_state[key] is not None:
                        if key == "analysis" and hasattr(final_state[key], "model_dump"):
                            completion_data[key] = final_state[key].model_dump()
                        elif key == "quality" and hasattr(final_state[key], "model_dump"):
                            completion_data[key] = final_state[key].model_dump()
                        else:
                            completion_data[key] = final_state[key]

            # Serialize and send completion message
            serialized_completion = serialize_nested(completion_data)
            yield f"data: {json.dumps(serialized_completion, ensure_ascii=False)}\n\n"

        except Exception as e:
            logging.error(f"❌ Streaming failed: {str(e)}")
            error_data = {"type": "error", "message": str(e)[:100], "timestamp": datetime.now().isoformat(), "error_type": type(e).__name__}
            try:
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception as json_error:
                # Minimal fallback error message
                logging.error(f"❌ Failed to serialize error data: {str(json_error)}")
                yield f'data: {{"type": "error", "message": "Streaming failed", "timestamp": "{datetime.now().isoformat()}"}}\n\n'

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
