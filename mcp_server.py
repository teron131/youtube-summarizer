"""Standalone FastMCP server for YouTube scraping and summarization."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from typing import Literal

from fastmcp import FastMCP

from youtube_summarizer.prompts import get_gemini_summary_prompt, get_langchain_summary_prompt
from youtube_summarizer.scrapper import extract_transcript_text, has_transcript_provider_key
from youtube_summarizer.settings import get_settings
from youtube_summarizer.summarizer_gemini import summarize_video_async as summarize_video_gemini
from youtube_summarizer.summarizer_openrouter import summarize_video_async as summarize_video_openrouter
from youtube_summarizer.utils import clean_youtube_url, is_youtube_url

mcp = FastMCP("YouTube Summarizer MCP")

ProviderType = Literal["auto", "openrouter", "gemini"]
TargetLanguage = Literal["auto", "en", "zh"]


def _run(coro):
    return asyncio.run(coro)


def _processing_time(start_time: datetime) -> str:
    return f"{(datetime.now(UTC) - start_time).total_seconds():.1f}s"


def _validate_url(url: str) -> str:
    clean_url = url.strip()
    if not clean_url:
        raise ValueError("URL required")
    if not is_youtube_url(clean_url):
        raise ValueError("Invalid YouTube URL")
    return clean_youtube_url(clean_url)


def _resolve_target_language(target_language: TargetLanguage | None) -> TargetLanguage:
    return target_language or get_settings().default_target_language


def _resolve_provider(requested_provider: ProviderType, url: str) -> Literal["openrouter", "gemini"]:
    settings = get_settings()
    has_openrouter = settings.has_openrouter
    has_gemini = settings.has_gemini

    if requested_provider == "openrouter":
        if not has_openrouter:
            raise ValueError("Config missing: OPENROUTER_API_KEY")
        return "openrouter"

    if requested_provider == "gemini":
        if not has_gemini:
            raise ValueError("Config missing: GEMINI_API_KEY")
        return "gemini"

    if is_youtube_url(url) and has_gemini:
        return "gemini"
    if has_openrouter:
        return "openrouter"
    if has_gemini:
        return "gemini"

    raise ValueError("Config missing: OPENROUTER_API_KEY or GEMINI_API_KEY")


async def _summarize_with_provider(
    url: str,
    provider: Literal["openrouter", "gemini"],
    target_language: TargetLanguage | None,
):
    resolved_target_language = _resolve_target_language(target_language)
    if provider == "gemini":
        summary, metadata = await summarize_video_gemini(
            url,
            target_language=resolved_target_language,
        )
        if summary is None:
            raise ValueError("Gemini summarization returned no content")
        return summary, metadata

    transcript = await extract_transcript_text(url)
    summary = await summarize_video_openrouter(
        transcript,
        resolved_target_language,
    )
    return summary, None


def _build_metadata(
    usage_metadata: dict[str, int | float] | None,
    processing_time: str,
) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {"processing_time": processing_time}
    if usage_metadata:
        metadata.update(usage_metadata)
    return metadata


@mcp.tool
def health() -> dict:
    """Return MCP server health and provider key availability."""
    settings = get_settings()
    return {
        "status": "healthy",
        "message": f"{settings.api_title} MCP server is running",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.api_version,
        "environment": {
            "gemini_configured": settings.has_gemini,
            "openrouter_configured": settings.has_openrouter,
            "scrapecreators_configured": settings.has_scrapecreators,
            "supadata_configured": settings.has_supadata,
        },
    }


@mcp.tool
def config() -> dict:
    """Return non-secret runtime configuration."""
    settings = get_settings()
    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "available_models": {
            "gemini_summary_model": settings.gemini_summary_model,
            "openrouter_summary_model": settings.openrouter_summary_model,
        },
        "available_providers": ["auto", "gemini", "openrouter"],
        "supported_languages": {
            "auto": "Auto detect (fallback to English if unclear)",
            "en": "English",
            "zh": "Traditional Chinese (繁體中文)",
        },
        "default_provider": settings.default_provider,
        "default_summary_model": settings.openrouter_summary_model,
        "default_target_language": settings.default_target_language,
        "settings": settings.to_public_config(),
    }


@mcp.tool
def scrape(url: str) -> dict:
    """Fetch normalized transcript text from a YouTube URL."""
    start_time = datetime.now(UTC)
    normalized_url = _validate_url(url)

    if not has_transcript_provider_key():
        raise ValueError("Config missing: SCRAPECREATORS_API_KEY or SUPADATA_API_KEY")

    transcript = _run(extract_transcript_text(normalized_url))
    return {
        "status": "success",
        "message": "Video scraped successfully",
        "url": normalized_url,
        "transcript": transcript,
        "metadata": {
            "processing_time": _processing_time(start_time),
        },
    }


@mcp.tool
def summarize(
    url: str,
    provider: ProviderType = "auto",
    target_language: TargetLanguage = "auto",
) -> dict:
    """Generate summary from a YouTube URL with provider routing."""
    start_time = datetime.now(UTC)
    normalized_url = _validate_url(url)
    resolved_target_language = _resolve_target_language(target_language)
    resolved_provider = _resolve_provider(provider, normalized_url)

    summary, usage_metadata = _run(
        _summarize_with_provider(
            url=normalized_url,
            provider=resolved_provider,
            target_language=resolved_target_language,
        )
    )

    return {
        "status": "success",
        "message": f"Summary completed successfully via {resolved_provider}",
        "summary": summary.model_dump(),
        "metadata": _build_metadata(usage_metadata, _processing_time(start_time)),
        "iteration_count": 1,
        "target_language": resolved_target_language,
    }


@mcp.tool
def prompt_preview(target_language: TargetLanguage = "auto") -> dict:
    """Preview effective summarization prompts for debugging language behavior."""
    return {
        "gemini_prompt": get_gemini_summary_prompt(target_language=target_language),
        "openrouter_prompt": get_langchain_summary_prompt(target_language=target_language),
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "http":
        host = os.getenv("MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
