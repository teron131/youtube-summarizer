"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralized runtime configuration for the API and providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "YouTube Summarizer MCP"

    default_target_language: Literal["auto", "en", "zh"] = "auto"

    scrape_timeout_seconds: int = 60
    llm_timeout_seconds: int = 120
    task_timeout_seconds: float = 300.0

    scrapecreators_transcript_url: str = "https://api.scrapecreators.com/v1/youtube/video/transcript"
    supadata_transcript_url: str = "https://api.supadata.ai/v1/transcript"

    openrouter_summary_model: str = "x-ai/grok-4.1-fast"
    openrouter_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"

    gemini_summary_model: str = "gemini-3-flash-preview"
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "medium"

    mcp_auth_mode: Literal["none", "google_oauth"] = "none"
    mcp_server_base_url: str | None = Field(default=None)
    mcp_google_client_id: str | None = Field(default=None)
    mcp_google_client_secret: str | None = Field(default=None)
    mcp_google_required_scopes: str = "openid"

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

    @property
    def mcp_google_scopes(self) -> list[str]:
        return [scope for scope in self.mcp_google_required_scopes.split() if scope]

    def to_public_config(self) -> dict[str, str | int | float | bool]:
        """Return safe config values for API responses and diagnostics."""
        return {
            "api_title": self.api_title,
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
            "mcp_auth_mode": self.mcp_auth_mode,
            "mcp_server_base_url_set": bool(self.mcp_server_base_url),
            "openrouter_configured": self.has_openrouter,
            "gemini_configured": self.has_gemini,
            "scrapecreators_configured": self.has_scrapecreators,
            "supadata_configured": self.has_supadata,
            "mcp_auth_enabled": self.mcp_auth_mode != "none",
        }


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    optional_key_fields = (
        "openrouter_api_key",
        "gemini_api_key",
        "google_api_key",
        "scrapecreators_api_key",
        "supadata_api_key",
        "mcp_server_base_url",
        "mcp_google_client_id",
        "mcp_google_client_secret",
    )
    for field_name in optional_key_fields:
        current_value = getattr(settings, field_name)
        setattr(settings, field_name, _clean_optional(current_value))

    return settings
