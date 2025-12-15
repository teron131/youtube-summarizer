"""YouTube video transcript summarization using LangChain ReAct agent with structured output.

This is a lightweight single-file alternative to the full LangGraph workflow in summarizer.py.
It uses LangChain's ReAct agent instead of the multi-node self-checking pipeline.
"""

from collections.abc import Generator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from .openrouter import ChatOpenRouter
from .scrapper import YouTubeScrapperResult, scrap_youtube
from .utils import is_youtube_url, s2hk

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

ANALYSIS_MODEL = "x-ai/grok-4.1-fast"
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)


# ============================================================================
# Tools
# ============================================================================


@tool
def scrap_youtube_tool(youtube_url: str) -> str:
    """Scrape a YouTube video and return the transcript.

    Args:
        youtube_url: The YouTube video URL to scrape

    Returns:
        Parsed transcript text
    """
    result: YouTubeScrapperResult = scrap_youtube(youtube_url)
    if not result.has_transcript:
        raise ValueError("Video has no transcript")
    if not result.parsed_transcript:
        raise ValueError("Transcript is empty")
    return result.parsed_transcript


# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the analysis."""

    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")

    @field_validator("header", "summary")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("key_points")
    def convert_list_to_hk(cls, value: list[str]) -> list[str]:
        """Convert list fields to Traditional Chinese."""
        return [s2hk(item) for item in value]


class Analysis(BaseModel):
    """Complete analysis of video content."""

    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(
        description="Key insights and actionable takeaways for the audience",
        min_length=3,
        max_length=8,
    )
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(
        description="The most relevant keywords in the analysis worthy of highlighting",
        min_length=3,
        max_length=3,
    )
    target_language: str | None = Field(default=None, description="The language the content to be translated to")

    @field_validator("title", "summary")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("takeaways", "keywords")
    def convert_list_to_hk(cls, value: list[str]) -> list[str]:
        """Convert list fields to Traditional Chinese."""
        return [s2hk(item) for item in value]


# ============================================================================
# Agent Creation
# ============================================================================


def create_summarizer_agent(target_language: str | None = None):
    """Create a ReAct agent for summarizing video transcripts with structured output.

    Args:
        target_language: Optional target language for the analysis output

    Returns:
        Configured LangChain agent with structured output
    """
    llm = ChatOpenRouter(
        model=ANALYSIS_MODEL,
        temperature=0,
        reasoning_effort="medium",
    )

    system_prompt = "Analyze the transcript and create a comprehensive analysis with clear structure, key insights, and meaningful keywords. Avoid meta-language phrases."
    if target_language:
        system_prompt += f" Output the analysis in {target_language}."

    agent = create_agent(
        model=llm,
        tools=[scrap_youtube_tool],
        system_prompt=system_prompt,
        response_format=ToolStrategy(Analysis),
    )

    return agent


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_transcript(transcript_or_url: str) -> str:
    """Extract transcript from URL or return text directly."""
    if is_youtube_url(transcript_or_url):
        result: YouTubeScrapperResult = scrap_youtube(transcript_or_url)
        if not result.has_transcript:
            raise ValueError("Video has no transcript")
        if not result.parsed_transcript:
            raise ValueError("Transcript is empty")
        return result.parsed_transcript

    if not transcript_or_url or not transcript_or_url.strip():
        raise ValueError("Transcript cannot be empty")

    return transcript_or_url


# ============================================================================
# Public API
# ============================================================================


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Analysis:
    """Summarize YouTube video or text transcript using ReAct agent."""
    transcript = _extract_transcript(transcript_or_url)
    agent = create_summarizer_agent(target_language or TARGET_LANGUAGE)

    prompt = f"Analyze this transcript:\n\n{transcript}"
    response: dict = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    structured_response = response.get("structured_response")
    if structured_response is None:
        raise ValueError("Agent did not return structured response")

    analysis = Analysis.model_validate(structured_response)
    return analysis


def stream_summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Generator[Analysis, None, None]:
    """Stream the summarization process with progress updates."""
    transcript = _extract_transcript(transcript_or_url)
    agent = create_summarizer_agent(target_language or TARGET_LANGUAGE)

    prompt = f"Analyze this transcript:\n\n{transcript}"

    for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}):
        yield Analysis.model_validate(chunk.get("structured_response"))
