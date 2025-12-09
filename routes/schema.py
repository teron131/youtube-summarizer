"""
Request and Response models for the API
"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from youtube_summarizer.summarizer import Analysis, Quality

# ================================
# BASE MODELS
# ================================


class BaseResponse(BaseModel):
    """Base response model."""

    status: str = Field(description="Response status: success or error")
    message: str = Field(description="Human-readable message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ================================
# REQUEST MODELS
# ================================


class YouTubeRequest(BaseModel):
    """YouTube URL request."""

    url: str = Field(..., min_length=10, description="YouTube video URL")


class SummarizeRequest(BaseModel):
    """Summarization request."""

    content: str | None = Field(default=None, description="Content to analyze (YouTube URL or transcript text)")
    content_type: str = Field(default="url", pattern=r"^(url|transcript)$")

    # Model selection
    analysis_model: str = Field(default="google/gemini-2.5-pro", description="Model for analysis generation")
    quality_model: str = Field(default="google/gemini-2.5-flash", description="Model for quality evaluation")

    # Translation options
    target_language: str | None = Field(default=None, description="Target language for translation (None for auto-detect)")

    @model_validator(mode="after")
    def validate_content_based_on_type(self):
        """Validate content based on content_type."""
        if self.content_type == "transcript" and (self.content is None or self.content.strip() == ""):
            raise ValueError("Content is required when content_type is 'transcript'")
        elif self.content_type == "url" and (self.content is None or self.content.strip() == ""):
            raise ValueError("Valid URL is required when content_type is 'url'")
        return self


# ================================
# RESPONSE MODELS
# ================================


class ScrapResponse(BaseResponse):
    """Video scraping response."""

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
    """Summarization response."""

    analysis: Analysis
    quality: Quality | None = None
    processing_time: str
    iteration_count: int = Field(default=1)

    # Model metadata
    analysis_model: str = Field(description="Model used for analysis")
    quality_model: str = Field(description="Model used for quality evaluation")

    # Translation metadata
    target_language: str | None = Field(default=None, description="Target language used for translation")


class ConfigurationResponse(BaseResponse):
    """Configuration response with available options."""

    available_models: dict[str, str] = Field(description="Available models for selection")
    supported_languages: dict[str, str] = Field(description="Supported languages for translation")
    default_analysis_model: str = Field(description="Default analysis model")
    default_quality_model: str = Field(description="Default quality model")
    default_target_language: str = Field(description="Default target language")
