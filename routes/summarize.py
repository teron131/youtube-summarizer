"""AI summarization endpoints with streaming and non-streaming modes."""

import asyncio
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from routes.schema import SummarizeRequest, SummarizeResponse
from youtube_summarizer.scrapper import scrap_youtube
from youtube_summarizer.summarizer import (
    SummarizerOutput,
    SummarizerState,
    create_graph,
    stream_summarize_video,
)
from youtube_summarizer.utils import is_youtube_url, serialize_nested

from .errors import handle_exception, require_env_key
from .helpers import get_processing_time, parse_scraper_result, run_async_task

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Content required")

    if request.content_type == "url" and not is_youtube_url(request.content):
        raise HTTPException(status_code=400, detail="Valid YouTube URL required")

    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        require_env_key("OPENROUTER_API_KEY")

    start_time = datetime.now()

    try:
        transcript = request.content
        if request.content_type == "url":
            scrap_result = await run_async_task(scrap_youtube, request.content)
            parsed_data = parse_scraper_result(scrap_result)
            transcript = parsed_data.get("transcript") or request.content

        graph = create_graph()
        initial_state = SummarizerState(
            transcript=transcript,
            target_language=request.target_language,
        )
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
        raise handle_exception(e, "Analysis")


@router.post("/stream-summarize")
async def stream_summarize(request: SummarizeRequest):
    require_env_key("GEMINI_API_KEY")

    async def generate_stream():
        start_time = datetime.now()
        try:
            if not request.content or not request.content.strip():
                raise HTTPException(status_code=400, detail="Content required")

            content = request.content
            if request.content_type == "url":
                if not is_youtube_url(content):
                    raise HTTPException(status_code=400, detail="Valid YouTube URL required")

                logging.info(f"🔗 Scraping YouTube video: {content}")
                scrap_result = await run_async_task(scrap_youtube, content)
                parsed_data = parse_scraper_result(scrap_result)
                content = parsed_data.get("transcript") or content

                if content == request.content:
                    logging.info("🎯 No transcript available - sending YouTube URL to Gemini")
                else:
                    logging.info("📝 Using scraped transcript for analysis")

            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...', 'timestamp': datetime.now().isoformat()})}\n\n"
            await asyncio.sleep(0.01)

            chunk_count = 0
            final_state = None

            for state_chunk in stream_summarize_video(content, request.target_language):
                try:
                    chunk_dict = state_chunk.model_dump() if hasattr(state_chunk, "model_dump") else {}
                    final_state = chunk_dict

                    serialized_chunk = serialize_nested(chunk_dict)
                    serialized_chunk["timestamp"] = datetime.now().isoformat()
                    serialized_chunk["chunk_number"] = chunk_count

                    yield f"data: {json.dumps(serialized_chunk, ensure_ascii=False)}\n\n"
                    chunk_count += 1

                    if chunk_count % 10 == 0:
                        await asyncio.sleep(0.01)

                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    logging.warning(f"⚠️ Failed to serialize chunk {chunk_count}: {str(e)}")
                    chunk_count += 1

            completion_data = {
                "type": "complete",
                "message": "Analysis completed",
                "processing_time": get_processing_time(start_time),
                "total_chunks": chunk_count,
                "timestamp": datetime.now().isoformat(),
            }

            if final_state:
                for key in ["analysis", "quality", "iteration_count", "is_complete"]:
                    if (value := final_state.get(key)) is not None:
                        completion_data[key] = value

            yield f"data: {json.dumps(serialize_nested(completion_data), ensure_ascii=False)}\n\n"

        except Exception as e:
            logging.error(f"❌ Streaming failed: {str(e)}")
            error_data = {
                "type": "error",
                "message": str(e)[:100],
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )
