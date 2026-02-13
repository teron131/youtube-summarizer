"""Health check and configuration endpoints for API monitoring."""

from datetime import UTC, datetime

from fastapi import APIRouter

from routes.schema import ConfigurationResponse
from youtube_summarizer.settings import get_settings

router = APIRouter()

SETTINGS = get_settings()

AVAILABLE_MODELS = {
    "gemini_summary_model": SETTINGS.gemini_summary_model,
    "openrouter_summary_model": SETTINGS.openrouter_summary_model,
}
AVAILABLE_PROVIDERS = ["auto", "gemini", "openrouter"]
DEFAULT_PROVIDER = SETTINGS.default_provider
DEFAULT_TARGET_LANGUAGE = SETTINGS.default_target_language

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
        "version": SETTINGS.api_version,
        "name": SETTINGS.api_title,
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
        "message": f"{SETTINGS.api_title} is running",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": SETTINGS.api_version,
        "environment": {
            "gemini_configured": SETTINGS.has_gemini,
            "openrouter_configured": SETTINGS.has_openrouter,
            "scrapecreators_configured": SETTINGS.has_scrapecreators,
            "supadata_configured": SETTINGS.has_supadata,
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
        "default_summary_model": SETTINGS.openrouter_summary_model,
        "default_target_language": DEFAULT_TARGET_LANGUAGE,
        "settings": SETTINGS.to_public_config(),
    }
