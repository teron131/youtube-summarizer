"""Request and Response models for the API"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from youtube_summarizer.schemas import Summary


class BaseResponse(BaseModel):
    status: str = Field(description="Response status: success or error")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=10, description="YouTube video URL")


class SummarizeRequest(BaseModel):
    url: str = Field(
        min_length=10,
        description="YouTube video URL to summarize",
    )
    provider: Literal["auto", "openrouter", "gemini"] = Field(
        default="auto",
        description="LLM provider route. 'auto' resolves using configured keys.",
    )
    target_language: str | None = Field(
        default="en",
        description="Target language for translation (ISO language code)",
    )


class ScrapeResponse(BaseResponse):
    url: str | None = None
    transcript: str | None = None
    metadata: dict[str, str | int | float] | None = None


class SummarizeResponse(BaseResponse):
    summary: Summary
    metadata: dict[str, str | int | float] | None = None
    iteration_count: int = Field(default=1)
    target_language: str | None = Field(
        default=None,
        description="Target language used for translation",
    )


class ConfigurationResponse(BaseResponse):
    available_models: dict[str, str] = Field(description="Available models for selection")
    available_providers: list[str] = Field(description="Available provider routes")
    supported_languages: dict[str, str] = Field(
        description="Supported languages for translation",
    )
    default_provider: str = Field(description="Default provider route")
    default_summary_model: str = Field(description="Default summary model")
    default_target_language: str = Field(description="Default target language")
