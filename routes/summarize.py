"""AI summarization endpoints with streaming and non-streaming modes."""

import asyncio
from datetime import UTC, datetime
import json
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from youtube_summarizer.scrapper import extract_transcript_text, has_transcript_provider_key
from youtube_summarizer.summarizer import (
    SummarizerOutput,
    SummarizerState,
    create_graph,
    stream_summarize_video,
)
from youtube_summarizer.summarizer_lite import summarize_video
from youtube_summarizer.utils import is_youtube_url, serialize_nested

from .errors import handle_exception
from .helpers import get_processing_time, run_async_task
from .schema import SummarizeRequest, SummarizeResponse

router = APIRouter()

TRANSCRIPT_CONFIG_ERROR = "Config missing: SCRAPECREATORS_API_KEY or SUPADATA_API_KEY or FAL_KEY"
LLM_CONFIG_ERROR = "Config missing: OPENROUTER_API_KEY or GEMINI_API_KEY"


def _require_transcript_config() -> None:
    if has_transcript_provider_key() or os.getenv("FAL_KEY"):
        return
    raise HTTPException(status_code=500, detail=TRANSCRIPT_CONFIG_ERROR)


def _require_llm_config() -> None:
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return
    raise HTTPException(status_code=500, detail=LLM_CONFIG_ERROR)


def _validate_summary_request(request: SummarizeRequest) -> None:
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Content required")

    if request.content_type == "url":
        if not is_youtube_url(request.content):
            raise HTTPException(status_code=400, detail="Valid YouTube URL required")
        _require_transcript_config()


async def _resolve_transcript(request: SummarizeRequest) -> str:
    if request.content_type == "url":
        logging.info("🔗 Scraping YouTube video: %s", request.content)
        transcript = await run_async_task(extract_transcript_text, request.content)
        logging.info("📝 Using provider transcript for summary")
        return transcript
    return request.content


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    _validate_summary_request(request)
    _require_llm_config()

    start_time = datetime.now(UTC)

    try:
        transcript = await _resolve_transcript(request)

        if request.fast_mode:
            summary = summarize_video(transcript, request.target_language)
            processing_time = get_processing_time(start_time)
            return SummarizeResponse(
                status="success",
                message="Fast summary completed successfully",
                summary=summary,
                quality=None,
                processing_time=processing_time,
                iteration_count=1,
                target_language=request.target_language or "en",
                analysis_model=request.analysis_model,
                quality_model=None,
            )

        graph = create_graph()
        initial_state = SummarizerState(
            transcript=transcript,
            target_language=request.target_language,
        )
        result_dict = await run_async_task(graph.invoke, initial_state.model_dump())
        output = SummarizerOutput.model_validate(result_dict)

        return SummarizeResponse(
            status="success",
            message="Summary completed successfully",
            summary=output.summary,
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
        raise handle_exception(e, "Summary") from e


@router.post("/stream-summarize")
async def stream_summarize(request: SummarizeRequest):
    _validate_summary_request(request)
    _require_llm_config()

    async def generate_stream():
        start_time = datetime.now(UTC)
        try:
            content = await _resolve_transcript(request)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting summary...', 'timestamp': datetime.now(UTC).isoformat()})}\n\n"
            await asyncio.sleep(0.01)

            chunk_count = 0
            final_state = None

            for state_chunk in stream_summarize_video(content, request.target_language):
                try:
                    chunk_dict = state_chunk.model_dump() if hasattr(state_chunk, "model_dump") else {}
                    final_state = chunk_dict

                    serialized_chunk = serialize_nested(chunk_dict)
                    serialized_chunk["timestamp"] = datetime.now(UTC).isoformat()
                    serialized_chunk["chunk_number"] = chunk_count

                    yield f"data: {json.dumps(serialized_chunk, ensure_ascii=False)}\n\n"
                    chunk_count += 1

                    if chunk_count % 10 == 0:
                        await asyncio.sleep(0.01)

                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    logging.warning("⚠️ Failed to serialize chunk %s: %s", chunk_count, e)
                    chunk_count += 1

            completion_data = {
                "type": "complete",
                "message": "Summary completed",
                "processing_time": get_processing_time(start_time),
                "total_chunks": chunk_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if final_state:
                for key in ["summary", "quality", "iteration_count", "is_complete"]:
                    if (value := final_state.get(key)) is not None:
                        completion_data[key] = value
                if request.fast_mode:
                    completion_data["quality"] = None
                    completion_data["iteration_count"] = 1

            yield f"data: {json.dumps(serialize_nested(completion_data), ensure_ascii=False)}\n\n"

        except Exception as e:
            logging.error("❌ Streaming failed: %s", e)
            error_data = {
                "type": "error",
                "message": str(e)[:100],
                "timestamp": datetime.now(UTC).isoformat(),
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
