"""YouTube video transcript summarization using LangChain with LangGraph self-checking workflow."""

from collections.abc import Generator
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .llm_harness import ChatOpenRouter, TagRange, filter_content, tag_content, untag_content
from .scrapper import extract_transcript_text
from .utils import is_youtube_url, s2hk

# ============================================================================
# Configuration
# ============================================================================

ANALYSIS_MODEL = "x-ai/grok-4.1-fast"
QUALITY_MODEL = "x-ai/grok-4.1-fast"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)


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
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience", min_length=3, max_length=8)
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="The most relevant keywords in the analysis worthy of highlighting", min_length=3, max_length=3)
    target_language: str | None = Field(default=None, description="The language the content to be translated to")

    @field_validator("title", "summary")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("takeaways", "keywords")
    def convert_list_to_hk(cls, value: list[str]) -> list[str]:
        """Convert list fields to Traditional Chinese."""
        return [s2hk(item) for item in value]


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the analysis."""

    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    no_garbage: Rate = Field(
        description="Rate for no_garbage: The promotional and meaningless content such as cliche intros, outros, filler, sponsorships, and other irrelevant segments are effectively removed"
    )
    meta_language_avoidance: Rate = Field(description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc.")
    useful_keywords: Rate = Field(description="Rate for keywords: The keywords are useful for highlighting the analysis")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [
            self.completeness,
            self.structure,
            self.no_garbage,
            self.meta_language_avoidance,
            self.useful_keywords,
            self.correct_language,
        ]

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score based on Pass/Refine/Fail ratings."""
        aspects = self.all_aspects
        pass_count = sum(1 for a in aspects if a.rate == "Pass")
        refine_count = sum(1 for a in aspects if a.rate == "Refine")
        # Pass = 100%, Refine = 50%, Fail = 0%
        return int((pass_count * 100 + refine_count * 50) / len(aspects))

    @property
    def is_acceptable(self) -> bool:
        """Check if quality score meets minimum threshold."""
        return self.percentage_score >= MIN_QUALITY_SCORE


class GarbageIdentification(BaseModel):
    """List of identified garbage sections in a transcript."""

    garbage_ranges: list[TagRange] = Field(description="List of line ranges identified as promotional or irrelevant content")


class SummarizerState(BaseModel):
    """State schema for the summarization graph."""

    transcript: str | None = None
    analysis: Analysis | None = None
    quality: Quality | None = None
    target_language: str | None = None
    iteration_count: int = 0
    is_complete: bool = False


class SummarizerOutput(BaseModel):
    """Output schema for the summarization graph."""

    analysis: Analysis
    quality: Quality | None = None
    iteration_count: int
    transcript: str | None = None


# ============================================================================
# Graph Nodes
# ============================================================================


def garbage_filter_node(state: SummarizerState) -> dict:
    """Identify and remove garbage from the transcript."""
    # Tag the transcript for identification
    tagged_transcript = tag_content(state.transcript)

    llm = ChatOpenRouter(
        model=FAST_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(GarbageIdentification)

    system_prompt = "Identify and remove garbage sections such as promotional and meaningless content such as cliche intros, outros, filler, sponsorships, and other irrelevant segments from the transcript. The transcript has line tags like [L1], [L2], etc. Return the ranges of tags that should be removed to clean the transcript."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=tagged_transcript),
    ]

    garbage: GarbageIdentification = llm.invoke(messages)

    if garbage.garbage_ranges:
        filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
        # Untag for the next stage (analysis)
        cleaned_transcript = untag_content(filtered_transcript)
        print(f"🧹 Removed {len(garbage.garbage_ranges)} garbage sections.")
        return {"transcript": cleaned_transcript}

    return {}


def analysis_node(state: SummarizerState) -> dict:
    """Generate analysis from transcript."""
    llm = ChatOpenRouter(
        model=ANALYSIS_MODEL,
        temperature=0,
        reasoning_effort="medium",
    ).with_structured_output(Analysis)

    system_prompt = "Analyze the transcript and create a comprehensive analysis with clear structure, key insights, and meaningful keywords. Avoid meta-language phrases."
    if state.target_language:
        system_prompt += f" Output the analysis in {state.target_language}."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}"),
    ]

    analysis = llm.invoke(messages)

    return {
        "analysis": analysis,
        "iteration_count": state.iteration_count + 1,
    }


def quality_node(state: SummarizerState) -> dict:
    """Assess quality of analysis."""
    llm = ChatOpenRouter(
        model=QUALITY_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(Quality)

    system_prompt = "Evaluate the analysis against the transcript and provide each aspect a rating (Pass/Refine/Fail) with reasons."
    if state.target_language:
        system_prompt += f" Verify that the analysis is in {state.target_language}."

    analysis_json = state.analysis.model_dump_json() if state.analysis else "No analysis provided"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}\n\nAnalysis:\n{analysis_json}"),
    ]

    quality: Quality = llm.invoke(messages)

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def should_continue(state: SummarizerState) -> str:
    """Determine next step in workflow."""
    quality_percent = state.quality.percentage_score if state.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"

    if state.is_complete:
        print(f"✅ Complete: quality {quality_display}")
        return END

    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Refining: quality {quality_display} < {MIN_QUALITY_SCORE}% (iteration {state.iteration_count + 1})")
        return "analysis"

    print(f"⚠️ Stopping: quality {quality_display}, {state.iteration_count} iterations")
    return END


def create_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(
        SummarizerState,
        output_schema=SummarizerOutput,
    )

    builder.add_node("garbage_filter", garbage_filter_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("quality", quality_node)

    builder.add_edge(START, "garbage_filter")
    builder.add_edge("garbage_filter", "analysis")
    builder.add_edge("analysis", "quality")

    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "analysis": "analysis",
            END: END,
        },
    )

    return builder.compile()


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
    """Summarize YouTube video or text transcript with quality self-checking."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    result: dict = graph.invoke(initial_state.model_dump())
    output = SummarizerOutput.model_validate(result)

    quality_percent = output.quality.percentage_score if output.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"
    print(f"🎯 Final: quality {quality_display}, {output.iteration_count} iterations")
    return output.analysis


def stream_summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Generator[SummarizerState, None, None]:
    """Stream the summarization process with progress updates."""
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    for chunk in graph.stream(initial_state.model_dump(), stream_mode="values"):
        yield SummarizerState.model_validate(chunk)
