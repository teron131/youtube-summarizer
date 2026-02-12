"""YouTube video transcript summarization using LangChain ReAct agent with structured output.

This is a lightweight single-file alternative to the full LangGraph workflow in summarizer.py.
It uses LangChain's ReAct agent instead of the multi-node self-checking pipeline.
"""

from collections.abc import Callable, Generator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from .fast_copy import (
    TagRange,
    filter_content,
    tag_content,
    untag_content,
)
from .openrouter import ChatOpenRouter
from .scrapper import extract_transcript_text
from .utils import is_youtube_url, s2hk

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

ANALYSIS_MODEL = "x-ai/grok-4.1-fast"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
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
    return extract_transcript_text(youtube_url)


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


class GarbageIdentification(BaseModel):
    """List of identified garbage sections in a content block."""

    garbage_ranges: list[TagRange] = Field(description="List of line ranges identified as promotional or irrelevant content")


@wrap_tool_call
def garbage_filter_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Middleware to filter garbage from tool results (like transcripts)."""
    result = handler(request)

    # Only filter if it's the scrap_youtube_tool and the call succeeded
    if request.tool_call["name"] == "scrap_youtube_tool" and result.status != "error":
        transcript = result.content
        if isinstance(transcript, str) and transcript.strip():
            # Apply the tagging/filtering mechanism
            tagged_transcript = tag_content(transcript)

            llm = ChatOpenRouter(
                model=FAST_MODEL,
                temperature=0,
            ).with_structured_output(GarbageIdentification)

            system_prompt = "Identify and remove garbage sections such as promotional and meaningless content such as cliche intros, outros, filler, sponsorships, and other irrelevant segments from the transcript. The transcript has line tags like [L1], [L2], etc. Return the ranges of tags that should be removed to clean the transcript."

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=tagged_transcript),
            ]

            garbage: GarbageIdentification = llm.invoke(messages)

            if garbage.garbage_ranges:
                filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
                cleaned_transcript = untag_content(filtered_transcript)
                print(f"🧹 Middleware removed {len(garbage.garbage_ranges)} garbage sections from tool result.")
                # Update the result content
                result.content = cleaned_transcript

    return result


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
        middleware=[garbage_filter_middleware],
        response_format=ToolStrategy(Analysis),
    )

    return agent


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_transcript(transcript_or_url: str) -> str:
    """Extract transcript from URL or return text directly."""
    if is_youtube_url(transcript_or_url):
        return extract_transcript_text(transcript_or_url)

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
