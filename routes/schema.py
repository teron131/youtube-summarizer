"""Request and Response models for the API"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from youtube_summarizer.schemas import Quality, Summary


class BaseResponse(BaseModel):
    status: str = Field(description="Response status: success or error")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=10, description="YouTube video URL")


class SummarizeRequest(BaseModel):
    content: str = Field(
        description="Content to analyze (YouTube URL or transcript text)",
    )
    target_language: str | None = Field(
        default="en",
        description="Target language for translation (ISO language code)",
    )
    fast_mode: bool = Field(default=False, description="Use fast summarization without quality checks")


class ScrapeResponse(BaseResponse):
    url: str | None = None
    transcript: str | None = None
    processing_time: str


# Backward-compatible alias.
ScrapResponse = ScrapeResponse


class SummarizeResponse(BaseResponse):
    summary: Summary
    quality: Quality | None = None
    processing_time: str
    iteration_count: int = Field(default=1)
    target_language: str | None = Field(
        default=None,
        description="Target language used for translation",
    )


class ConfigurationResponse(BaseResponse):
    available_models: dict[str, str] = Field(description="Available models for selection")
    supported_languages: dict[str, str] = Field(
        description="Supported languages for translation",
    )
    default_summary_model: str = Field(description="Default summary model")
    default_quality_model: str = Field(description="Default quality model")
    default_target_language: str = Field(description="Default target language")
