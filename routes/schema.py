"""Request and Response models for the API"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator
from youtube_summarizer.summarizer import Analysis, Quality


class BaseResponse(BaseModel):
    status: str = Field(description="Response status: success or error")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=10, description="YouTube video URL")


class SummarizeRequest(BaseModel):
    content: str | None = Field(
        default=None,
        description="Content to analyze (YouTube URL or transcript text)",
    )
    content_type: str = Field(default="url", pattern=r"^(url|transcript)$")
    analysis_model: str = Field(
        default="x-ai/grok-4.1-fast",
        description="Model for analysis generation",
    )
    quality_model: str = Field(
        default="x-ai/grok-4.1-fast",
        description="Model for quality evaluation",
    )
    target_language: str | None = Field(
        default="en",
        description="Target language for translation (ISO language code)",
    )

    @model_validator(mode="after")
    def validate_content_based_on_type(self):
        if not self.content or not self.content.strip():
            content_label = "Content" if self.content_type == "transcript" else "Valid URL"
            raise ValueError(f"{content_label} is required when content_type is '{self.content_type}'")
        return self


class ScrapResponse(BaseResponse):
    url: str | None = None
    title: str | None = None
    author: str | None = None
    transcript: str | None = None
    duration: str | None = None
    thumbnail: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    upload_date: str | None = None
    processing_time: str


class SummarizeResponse(BaseResponse):
    analysis: Analysis
    quality: Quality | None = None
    processing_time: str
    iteration_count: int = Field(default=1)
    analysis_model: str = Field(description="Model used for analysis")
    quality_model: str = Field(description="Model used for quality evaluation")
    target_language: str | None = Field(
        default=None,
        description="Target language used for translation",
    )


class ConfigurationResponse(BaseResponse):
    available_models: dict[str, str] = Field(description="Available models for selection")
    supported_languages: dict[str, str] = Field(
        description="Supported languages for translation",
    )
    default_analysis_model: str = Field(description="Default analysis model")
    default_quality_model: str = Field(description="Default quality model")
    default_target_language: str = Field(description="Default target language")
