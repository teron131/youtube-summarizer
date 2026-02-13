"""Health check and configuration endpoints for API monitoring."""

from datetime import UTC, datetime
import os

from fastapi import APIRouter

from routes.schema import ConfigurationResponse
from youtube_summarizer.scrapper.supadata import get_supadata_api_key
from youtube_summarizer.summarizer_gemini import GEMINI_SUMMARY_MODEL
from youtube_summarizer.summarizer_lite import OPENROUTER_FILTER_MODEL, OPENROUTER_SUMMARY_MODEL

router = APIRouter()

API_VERSION = "3.0.0"
API_TITLE = "YouTube Summarizer API"

AVAILABLE_MODELS = {
    "gemini_summary_model": GEMINI_SUMMARY_MODEL,
    "openrouter_summary_model": OPENROUTER_SUMMARY_MODEL,
    "openrouter_filter_model": OPENROUTER_FILTER_MODEL,
}
AVAILABLE_PROVIDERS = ["auto", "gemini", "openrouter"]
DEFAULT_PROVIDER = "auto"
DEFAULT_TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "en")

SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "ru": "Russian",
}


@router.get("/")
async def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "Optimized YouTube video processing and summarization",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check with environment status",
            "GET /config": "Get available models and languages",
            "POST /scrape": "Extract video metadata and transcript",
            "POST /summarize": "Provider-based summary generation",
            "POST /stream-summarize": "Streaming summary events",
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": f"{API_TITLE} is running",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": API_VERSION,
        "environment": {
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY")),
            "scrapecreators_configured": bool(os.getenv("SCRAPECREATORS_API_KEY")),
            "supadata_configured": bool(get_supadata_api_key()),
        },
    }


@router.get("/config", response_model=ConfigurationResponse)
async def get_configuration():
    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "available_models": AVAILABLE_MODELS,
        "available_providers": AVAILABLE_PROVIDERS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "default_provider": DEFAULT_PROVIDER,
        "default_summary_model": OPENROUTER_SUMMARY_MODEL,
        "default_quality_model": OPENROUTER_FILTER_MODEL,
        "default_target_language": DEFAULT_TARGET_LANGUAGE,
    }
