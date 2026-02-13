"""AI summarization endpoints with streaming and non-streaming modes."""

import asyncio
from datetime import UTC, datetime
import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from youtube_summarizer.schemas import Summary
from youtube_summarizer.scrapper import extract_transcript_text
from youtube_summarizer.settings import get_settings
from youtube_summarizer.summarizer_gemini import summarize_video_async as summarize_video_gemini
from youtube_summarizer.summarizer_openrouter import summarize_video_async as summarize_video_openrouter
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

from .errors import handle_exception
from .helpers import get_processing_time
from .schema import SummarizeRequest, SummarizeResponse

router = APIRouter()

LLM_CONFIG_ERROR = "Config missing: OPENROUTER_API_KEY or GEMINI_API_KEY"
ProviderType = Literal["auto", "openrouter", "gemini"]


def _require_any_llm_config() -> None:
    settings = get_settings()
    if settings.has_any_llm:
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
    settings = get_settings()
    has_openrouter = settings.has_openrouter
    has_gemini = settings.has_gemini

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
) -> tuple[Summary, dict[str, int | float] | None]:
    if provider == "gemini":
        summary, metadata = await summarize_video_gemini(
            url,
            target_language=target_language or get_settings().default_target_language,
        )
        if summary is None:
            raise ValueError("Gemini summarization returned no content")
        return summary, metadata

    logging.info("🔗 Scraping YouTube video for OpenRouter: %s", url)
    transcript = await extract_transcript_text(url)
    logging.info("📝 Using provider transcript for OpenRouter summary")

    summary = await summarize_video_openrouter(
        transcript,
        target_language,
    )
    return summary, None


def _build_metadata(
    usage_metadata: dict[str, int | float] | None,
    processing_time: str,
) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
        "processing_time": processing_time,
    }
    if usage_metadata:
        metadata.update(usage_metadata)
    return metadata


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    url = _validate_summary_request(request)
    _require_any_llm_config()

    start_time = datetime.now(UTC)

    try:
        provider = _resolve_provider(request.provider, url)
        summary, metadata = await _summarize_with_provider(
            url=url,
            provider=provider,
            target_language=request.target_language,
        )

        return SummarizeResponse(
            status="success",
            message=f"Summary completed successfully via {provider}",
            summary=summary,
            metadata=_build_metadata(metadata, get_processing_time(start_time)),
            iteration_count=1,
            target_language=request.target_language or get_settings().default_target_language,
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

            summary, metadata = await _summarize_with_provider(
                url=url,
                provider=provider,
                target_language=request.target_language,
            )
            completion_data = {
                "type": "complete",
                "message": f"Summary completed via {provider}",
                "total_chunks": 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "provider": provider,
                "summary": summary.model_dump(),
                "metadata": _build_metadata(metadata, get_processing_time(start_time)),
                "iteration_count": 1,
                "target_language": request.target_language or get_settings().default_target_language,
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
