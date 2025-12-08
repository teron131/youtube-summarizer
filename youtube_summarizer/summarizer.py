"""YouTube video transcript summarization using LangChain with LangGraph self-checking workflow."""

from typing import Generator, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .openrouter import ChatOpenRouter
from .utils import is_youtube_url, s2hk
from .youtube_scrapper import YouTubeScrapperResult, scrap_youtube

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

ANALYSIS_MODEL = "x-ai/grok-4.1-fast"
QUALITY_MODEL = "x-ai/grok-4.1-fast"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)

# Gemini-specific configuration
GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"
GEMINI_QUALITY_MODEL = "gemini-2.5-flash"

# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the analysis."""

    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")

    @field_validator("header", "summary")
    @classmethod
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("key_points")
    @classmethod
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
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")

    @field_validator("title", "summary")
    @classmethod
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    @field_validator("takeaways", "keywords")
    @classmethod
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
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
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


class SummarizerState(BaseModel):
    """State schema for the summarization graph."""

    transcript: Optional[str] = None
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None
    target_language: Optional[str] = None
    iteration_count: int = 0
    is_complete: bool = False


class SummarizerOutput(BaseModel):
    """Output schema for the summarization graph."""

    analysis: Analysis
    quality: Optional[Quality] = None
    iteration_count: int
    transcript: Optional[str] = None


# ============================================================================
# Graph Nodes
# ============================================================================


def langchain_analysis_node(state: SummarizerState) -> dict:
    """Generate analysis from transcript using LangChain."""
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
    if state.target_language:
        analysis.target_language = state.target_language

    return {
        "analysis": analysis,
        "iteration_count": state.iteration_count + 1,
    }


def gemini_analysis_node(state: SummarizerState) -> dict:
    """Generate analysis from transcript using Gemini via ChatOpenRouter."""
    llm = ChatOpenRouter(
        model=GEMINI_DEFAULT_MODEL,
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
    if state.target_language:
        analysis.target_language = state.target_language

    return {
        "analysis": analysis,
        "iteration_count": state.iteration_count + 1,
    }


def langchain_quality_node(state: SummarizerState) -> dict:
    """Assess quality of analysis using LangChain."""
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
    quality_percent = quality.percentage_score
    print(f"📈 Quality score: {quality_percent}%")

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable or state.iteration_count >= MAX_ITERATIONS,
    }


def gemini_quality_node(state: SummarizerState) -> dict:
    """Assess quality of analysis using Gemini via ChatOpenRouter."""
    llm = ChatOpenRouter(
        model=GEMINI_QUALITY_MODEL,
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
    quality_percent = quality.percentage_score
    print(f"📈 Quality score: {quality_percent}%")

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable or state.iteration_count >= MAX_ITERATIONS,
    }


# ============================================================================
# Graph Construction
# ============================================================================


def langchain_or_gemini(state: SummarizerState) -> str:
    """Determine whether to use Gemini SDK or LangChain based on input type."""
    # Use Gemini for YouTube URLs (after transcript extraction), LangChain for text
    # Since transcript is already extracted, we check the original input
    # For now, default to LangChain unless we have a way to track original input type
    # In practice, you might want to add a flag to state or check transcript characteristics
    return "langchain_analysis"


def should_continue_langchain(state: SummarizerState) -> str:
    """Determine next step in LangChain workflow."""
    quality_percent = state.quality.percentage_score if state.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"

    if state.is_complete:
        print(f"✅ Complete: quality {quality_display}")
        return END
    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Refining: quality {quality_display} < {MIN_QUALITY_SCORE}% (iteration {state.iteration_count + 1})")
        return "langchain_analysis"
    print(f"⚠️ Stopping: quality {quality_display}, {state.iteration_count} iterations")
    return END


def should_continue_gemini(state: SummarizerState) -> str:
    """Determine next step in Gemini workflow."""
    quality_percent = state.quality.percentage_score if state.quality else None
    quality_display = f"{quality_percent}%" if quality_percent is not None else "N/A"

    if state.is_complete:
        print(f"✅ Complete: quality {quality_display}")
        return END
    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Refining: quality {quality_display} < {MIN_QUALITY_SCORE}% (iteration {state.iteration_count + 1})")
        return "gemini_analysis"
    print(f"⚠️ Stopping: quality {quality_display}, {state.iteration_count} iterations")
    return END


def create_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(
        SummarizerState,
        output_schema=SummarizerOutput,
    )

    builder.add_node("langchain_analysis", langchain_analysis_node)
    builder.add_node("langchain_quality", langchain_quality_node)
    builder.add_node("gemini_analysis", gemini_analysis_node)
    builder.add_node("gemini_quality", gemini_quality_node)

    # Add conditional routing from START
    builder.add_conditional_edges(
        START,
        langchain_or_gemini,
        {
            "langchain_analysis": "langchain_analysis",
            "gemini_analysis": "gemini_analysis",
        },
    )

    # Add edges from analysis nodes to quality nodes
    builder.add_edge("langchain_analysis", "langchain_quality")
    builder.add_edge("gemini_analysis", "gemini_quality")

    # Add conditional edges from quality nodes
    builder.add_conditional_edges(
        "langchain_quality",
        should_continue_langchain,
        {
            "langchain_analysis": "langchain_analysis",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "gemini_quality",
        should_continue_gemini,
        {
            "gemini_analysis": "gemini_analysis",
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
    target_language: Optional[str] = None,
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
    target_language: Optional[str] = None,
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
