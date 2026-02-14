"""Standalone FastMCP server for YouTube scraping and summarization."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Literal

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider
from fastmcp.server.auth.providers.google import GoogleProvider

from youtube_summarizer import (
    AppSettings,
    Summary,
    clean_youtube_url,
    extract_transcript_text,
    get_settings,
    has_transcript_provider_key,
    is_youtube_url,
    summarize_video_gemini,
    summarize_video_openrouter,
)

TargetLanguage = Literal["auto", "en", "zh"]


def _build_auth_provider(settings: AppSettings) -> AuthProvider | None:
    if settings.mcp_auth_mode == "none":
        return None

    if settings.mcp_auth_mode == "google_oauth":
        if not settings.mcp_server_base_url:
            raise ValueError("MCP_SERVER_BASE_URL is required when MCP_AUTH_MODE=google_oauth")
        if not settings.mcp_google_client_id:
            raise ValueError("MCP_GOOGLE_CLIENT_ID is required when MCP_AUTH_MODE=google_oauth")
        if not settings.mcp_google_client_secret:
            raise ValueError("MCP_GOOGLE_CLIENT_SECRET is required when MCP_AUTH_MODE=google_oauth")

        return GoogleProvider(
            client_id=settings.mcp_google_client_id,
            client_secret=settings.mcp_google_client_secret,
            base_url=settings.mcp_server_base_url,
            required_scopes=settings.mcp_google_scopes,
        )

    raise ValueError(f"Unsupported MCP_AUTH_MODE: {settings.mcp_auth_mode}")


mcp = FastMCP(
    "YouTube Summarizer MCP",
    auth=_build_auth_provider(get_settings()),
    streamable_http_path="/",
)


def _processing_time(start_time: datetime) -> str:
    return f"{(datetime.now(UTC) - start_time).total_seconds():.1f}s"


def _validate_url(url: str) -> str:
    clean_url = url.strip()
    if not clean_url:
        raise ValueError("URL required")
    if not is_youtube_url(clean_url):
        raise ValueError("Invalid YouTube URL")
    return clean_youtube_url(clean_url)


def _resolve_target_language() -> TargetLanguage:
    return get_settings().default_target_language


def _resolve_provider() -> Literal["openrouter", "gemini"]:
    settings = get_settings()
    has_openrouter = settings.has_openrouter
    has_gemini = settings.has_gemini

    if has_gemini:
        return "gemini"
    if has_openrouter:
        return "openrouter"

    raise ValueError("Config missing: OPENROUTER_API_KEY or GEMINI_API_KEY")


async def _summarize_with_provider(
    url: str,
    provider: Literal["openrouter", "gemini"],
    target_language: TargetLanguage,
) -> tuple[Summary, dict[str, int | float] | None]:
    if provider == "gemini":
        summary, metadata = await summarize_video_gemini(
            url,
            target_language=target_language,
        )
        if summary is None:
            raise ValueError("Gemini summarization returned no content")
        return summary, metadata

    transcript = await extract_transcript_text(url)
    summary = await summarize_video_openrouter(
        transcript,
        target_language,
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
        "environment": {
            "gemini_configured": settings.has_gemini,
            "openrouter_configured": settings.has_openrouter,
            "scrapecreators_configured": settings.has_scrapecreators,
            "supadata_configured": settings.has_supadata,
        },
    }


@mcp.tool
async def scrape(url: str) -> dict:
    """Fetch normalized transcript text from a YouTube URL."""
    start_time = datetime.now(UTC)
    normalized_url = _validate_url(url)

    if not has_transcript_provider_key():
        raise ValueError("Config missing: SCRAPECREATORS_API_KEY or SUPADATA_API_KEY")

    transcript = await extract_transcript_text(normalized_url)
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
async def summarize(
    url: str,
) -> dict:
    """Generate summary from a YouTube URL using internal provider fallback."""
    start_time = datetime.now(UTC)
    normalized_url = _validate_url(url)
    resolved_target_language = _resolve_target_language()
    resolved_provider = _resolve_provider()

    summary, usage_metadata = await _summarize_with_provider(
        url=normalized_url,
        provider=resolved_provider,
        target_language=resolved_target_language,
    )

    return {
        "status": "success",
        "message": f"Summary completed successfully via {resolved_provider}",
        "summary": summary.model_dump(),
        "metadata": _build_metadata(usage_metadata, _processing_time(start_time)),
        "iteration_count": 1,
        "target_language": resolved_target_language,
    }


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")  # noqa: S104
        port = int(os.getenv("PORT") or os.getenv("MCP_PORT") or "8000")
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
