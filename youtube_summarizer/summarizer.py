"""YouTube video transcript summarization using LangChain with LangGraph self-checking workflow."""

from collections.abc import Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .llm_harness import ChatOpenRouter, filter_content, tag_content, untag_content
from .prompts import get_garbage_filter_prompt, get_langchain_summary_prompt, get_quality_check_prompt
from .schemas import GarbageIdentification, Quality, Summary
from .scrapper import extract_transcript_text
from .utils import is_youtube_url

SUMMARY_MODEL = "x-ai/grok-4.1-fast"
ANALYSIS_MODEL = SUMMARY_MODEL  # Backward-compatible alias used by existing routes.
QUALITY_MODEL = "x-ai/grok-4.1-fast"
FAST_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2
TARGET_LANGUAGE = "en"


class SummarizerState(BaseModel):
    transcript: str | None = None
    summary: Summary | None = None
    quality: Quality | None = None
    target_language: str | None = None
    iteration_count: int = 0
    is_complete: bool = False


class SummarizerOutput(BaseModel):
    summary: Summary
    quality: Quality | None = None
    iteration_count: int
    transcript: str | None = None


def garbage_filter_node(state: SummarizerState) -> dict:
    tagged_transcript = tag_content(state.transcript)
    llm = ChatOpenRouter(
        model=FAST_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(GarbageIdentification)

    messages = [
        SystemMessage(content=get_garbage_filter_prompt()),
        HumanMessage(content=tagged_transcript),
    ]
    garbage: GarbageIdentification = llm.invoke(messages)

    if not garbage.garbage_ranges:
        return {}

    filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
    cleaned_transcript = untag_content(filtered_transcript)
    return {"transcript": cleaned_transcript}


def summary_node(state: SummarizerState) -> dict:
    llm = ChatOpenRouter(
        model=SUMMARY_MODEL,
        temperature=0,
        reasoning_effort="medium",
    ).with_structured_output(Summary)

    messages = [
        SystemMessage(content=get_langchain_summary_prompt(target_language=state.target_language)),
        HumanMessage(content=f"Transcript:\n{state.transcript}"),
    ]
    summary = llm.invoke(messages)

    return {
        "summary": summary,
        "iteration_count": state.iteration_count + 1,
    }


def quality_node(state: SummarizerState) -> dict:
    llm = ChatOpenRouter(
        model=QUALITY_MODEL,
        temperature=0,
        reasoning_effort="low",
    ).with_structured_output(Quality)

    summary_json = state.summary.model_dump_json() if state.summary else "No summary provided"
    messages = [
        SystemMessage(content=get_quality_check_prompt(target_language=state.target_language)),
        HumanMessage(content=f"Transcript:\n{state.transcript}\n\nSummary:\n{summary_json}"),
    ]
    quality: Quality = llm.invoke(messages)

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable,
    }


def should_continue(state: SummarizerState) -> str:
    if state.is_complete:
        return END

    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        return "summary"

    return END


def create_graph() -> StateGraph:
    builder = StateGraph(
        SummarizerState,
        output_schema=SummarizerOutput,
    )
    builder.add_node("garbage_filter", garbage_filter_node)
    builder.add_node("summary", summary_node)
    builder.add_node("quality", quality_node)

    builder.add_edge(START, "garbage_filter")
    builder.add_edge("garbage_filter", "summary")
    builder.add_edge("summary", "quality")

    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "summary": "summary",
            END: END,
        },
    )
    return builder.compile()


def _extract_transcript(transcript_or_url: str) -> str:
    if is_youtube_url(transcript_or_url):
        return extract_transcript_text(transcript_or_url)

    if not transcript_or_url or not transcript_or_url.strip():
        raise ValueError("Transcript cannot be empty")

    return transcript_or_url


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    result: dict = graph.invoke(initial_state.model_dump())
    output = SummarizerOutput.model_validate(result)
    return output.summary


def stream_summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Generator[SummarizerState, None, None]:
    graph = create_graph()
    transcript = _extract_transcript(transcript_or_url)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language or TARGET_LANGUAGE,
    )
    for chunk in graph.stream(initial_state.model_dump(), stream_mode="values"):
        yield SummarizerState.model_validate(chunk)
