"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralized runtime configuration for the API and providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "YouTube Summarizer API"
    api_version: str = "3.0.0"

    default_provider: str = "auto"
    default_target_language: str = "en"

    scrape_timeout_seconds: int = 60
    llm_timeout_seconds: int = 120
    task_timeout_seconds: float = 300.0

    scrapecreators_transcript_url: str = "https://api.scrapecreators.com/v1/youtube/video/transcript"
    supadata_transcript_url: str = "https://api.supadata.ai/v1/transcript"

    openrouter_summary_model: str = "x-ai/grok-4.1-fast"
    openrouter_reasoning_effort: str = "medium"

    gemini_summary_model: str = "gemini-3-flash-preview"
    gemini_thinking_level: str = "medium"

    openrouter_api_key: str | None = Field(default=None)
    gemini_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)
    scrapecreators_api_key: str | None = Field(default=None)
    supadata_api_key: str | None = Field(default=None)

    @property
    def llm_timeout_milliseconds(self) -> int:
        """google-genai HttpOptions.timeout expects milliseconds."""
        return self.llm_timeout_seconds * 1000

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())

    @property
    def has_gemini(self) -> bool:
        return bool((self.gemini_api_key and self.gemini_api_key.strip()) or (self.google_api_key and self.google_api_key.strip()))

    @property
    def has_scrapecreators(self) -> bool:
        return bool(self.scrapecreators_api_key and self.scrapecreators_api_key.strip())

    @property
    def has_supadata(self) -> bool:
        return bool(self.supadata_api_key and self.supadata_api_key.strip())

    @property
    def has_any_llm(self) -> bool:
        return self.has_openrouter or self.has_gemini

    @property
    def has_any_transcript_provider(self) -> bool:
        return self.has_scrapecreators or self.has_supadata

    def to_public_config(self) -> dict[str, str | int | float | bool]:
        """Return safe config values for API responses and diagnostics."""
        return {
            "api_title": self.api_title,
            "api_version": self.api_version,
            "default_provider": self.default_provider,
            "default_target_language": self.default_target_language,
            "scrape_timeout_seconds": self.scrape_timeout_seconds,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "task_timeout_seconds": self.task_timeout_seconds,
            "scrapecreators_transcript_url": self.scrapecreators_transcript_url,
            "supadata_transcript_url": self.supadata_transcript_url,
            "openrouter_summary_model": self.openrouter_summary_model,
            "openrouter_reasoning_effort": self.openrouter_reasoning_effort,
            "gemini_summary_model": self.gemini_summary_model,
            "gemini_thinking_level": self.gemini_thinking_level,
            "openrouter_configured": self.has_openrouter,
            "gemini_configured": self.has_gemini,
            "scrapecreators_configured": self.has_scrapecreators,
            "supadata_configured": self.has_supadata,
        }


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()

    settings.openrouter_api_key = _clean_optional(settings.openrouter_api_key)
    settings.gemini_api_key = _clean_optional(settings.gemini_api_key)
    settings.google_api_key = _clean_optional(settings.google_api_key)
    settings.scrapecreators_api_key = _clean_optional(settings.scrapecreators_api_key)
    settings.supadata_api_key = _clean_optional(settings.supadata_api_key)

    return settings
