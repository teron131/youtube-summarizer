"""AI summarization endpoints with streaming and non-streaming modes."""

import asyncio
from datetime import UTC, datetime
import json
import logging
import os
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from youtube_summarizer.scrapper import extract_transcript_text
from youtube_summarizer.summarizer_gemini import summarize_video as summarize_video_gemini
from youtube_summarizer.summarizer_lite import summarize_video as summarize_video_openrouter
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

from .errors import handle_exception
from .helpers import get_processing_time, run_async_task
from .schema import SummarizeRequest, SummarizeResponse

router = APIRouter()

LLM_CONFIG_ERROR = "Config missing: OPENROUTER_API_KEY or GEMINI_API_KEY"
ProviderType = Literal["auto", "openrouter", "gemini"]


def _require_any_llm_config() -> None:
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return
    raise HTTPException(status_code=500, detail=LLM_CONFIG_ERROR)


def _validate_summary_request(request: SummarizeRequest) -> str:
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL required")
    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return clean_youtube_url(url)


def _resolve_provider(requested_provider: ProviderType, url: str) -> Literal["openrouter", "gemini"]:
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if requested_provider == "openrouter":
        if not has_openrouter:
            raise HTTPException(status_code=500, detail="Config missing: OPENROUTER_API_KEY")
        return "openrouter"

    if requested_provider == "gemini":
        if not has_gemini:
            raise HTTPException(status_code=500, detail="Config missing: GEMINI_API_KEY")
        return "gemini"

    if is_youtube_url(url) and has_gemini:
        return "gemini"
    if has_openrouter:
        return "openrouter"
    if has_gemini:
        return "gemini"

    raise HTTPException(status_code=500, detail=LLM_CONFIG_ERROR)


async def _summarize_with_provider(
    url: str,
    provider: Literal["openrouter", "gemini"],
    target_language: str | None,
):
    if provider == "gemini":
        summary = await run_async_task(
            lambda: summarize_video_gemini(
                url,
                target_language=target_language or "en",
            )
        )
        if summary is None:
            raise ValueError("Gemini summarization returned no content")
        return summary

    logging.info("🔗 Scraping YouTube video for OpenRouter: %s", url)
    transcript = await run_async_task(extract_transcript_text, url)
    logging.info("📝 Using provider transcript for OpenRouter summary")

    return await run_async_task(
        summarize_video_openrouter,
        transcript,
        target_language,
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    url = _validate_summary_request(request)
    _require_any_llm_config()

    start_time = datetime.now(UTC)

    try:
        provider = _resolve_provider(request.provider, url)
        summary = await _summarize_with_provider(
            url=url,
            provider=provider,
            target_language=request.target_language,
        )

        return SummarizeResponse(
            status="success",
            message=f"Summary completed successfully via {provider}",
            summary=summary,
            quality=None,
            processing_time=get_processing_time(start_time),
            iteration_count=1,
            target_language=request.target_language or "en",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception(e, "Summary") from e


@router.post("/stream-summarize")
async def stream_summarize(request: SummarizeRequest):
    url = _validate_summary_request(request)
    _require_any_llm_config()

    async def generate_stream():
        start_time = datetime.now(UTC)
        try:
            provider = _resolve_provider(request.provider, url)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting summary...', 'timestamp': datetime.now(UTC).isoformat()})}\n\n"
            await asyncio.sleep(0.01)

            summary = await _summarize_with_provider(
                url=url,
                provider=provider,
                target_language=request.target_language,
            )
            completion_data = {
                "type": "complete",
                "message": f"Summary completed via {provider}",
                "processing_time": get_processing_time(start_time),
                "total_chunks": 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "provider": provider,
                "summary": summary.model_dump(),
                "quality": None,
                "iteration_count": 1,
                "target_language": request.target_language or "en",
            }
            yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"

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
