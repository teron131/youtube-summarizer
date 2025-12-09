"""YouTube Summarizer Actor - Main entry point.

This Actor analyzes YouTube videos using LangGraph workflow with quality self-checking.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Optional

import requests
from apify import Actor
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2


# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the analysis."""

    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")


class Analysis(BaseModel):
    """Complete analysis of video content."""

    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways", min_length=3, max_length=8)
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="Most relevant keywords", min_length=3, max_length=3)


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: str = Field(description="Score: Fail, Refine, or Pass")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the analysis."""

    completeness: Rate = Field(description="The entire transcript has been considered")
    structure: Rate = Field(description="The result is in desired structures")
    no_garbage: Rate = Field(description="Promotional and meaningless content are removed")
    meta_language_avoidance: Rate = Field(description="No phrases like 'This chapter introduces'")
    useful_keywords: Rate = Field(description="Keywords are useful for highlighting")
    correct_language: Rate = Field(description="Match original or requested language")

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


# ============================================================================
# YouTube Scraping
# ============================================================================


def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    youtube_patterns = [
        r"youtube\.com/watch\?v=",
        r"youtu\.be/",
        r"youtube\.com/embed/",
        r"youtube\.com/v/",
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def clean_youtube_url(url: str) -> str:
    """Clean and normalize a YouTube URL."""
    if "youtube.com/watch" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    elif "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def scrape_youtube_video(youtube_url: str, scrapecreators_api_key: str) -> dict:
    """Scrape YouTube video metadata and transcript using Scrape Creators API.

    Returns:
        dict with keys: title, author, transcript, duration, thumbnail, view_count, like_count
    """
    if not is_youtube_url(youtube_url):
        raise ValueError("Invalid YouTube URL")

    youtube_url = clean_youtube_url(youtube_url)

    url = f"https://api.scrapecreators.com/v1/youtube/video?url={youtube_url}&get_transcript=true"
    headers = {"x-api-key": scrapecreators_api_key}
    response = requests.get(url, headers=headers, timeout=60)  # Increased timeout
    response.raise_for_status()

    data = response.json()

    # Debug: Log response structure
    Actor.log.info(f"API Response keys: {list(data.keys())}")
    Actor.log.info(f"Has transcript field: {data.get('transcript') is not None}")
    Actor.log.info(f"Has transcript_only_text field: {data.get('transcript_only_text') is not None}")

    # Extract transcript text
    transcript_text = data.get("transcript_only_text", "")
    if not transcript_text or not transcript_text.strip():
        # Log more details for debugging
        transcript_list = data.get("transcript")
        if transcript_list and isinstance(transcript_list, list) and len(transcript_list) > 0:
            # Try to manually construct transcript from segments
            Actor.log.info(f"Found {len(transcript_list)} transcript segments, constructing text...")
            transcript_text = " ".join(segment.get("text", "") for segment in transcript_list if segment.get("text"))

        if not transcript_text or not transcript_text.strip():
            raise ValueError(f"Video has no transcript available. " f"Response had {len(data.keys())} fields. " f"Try a video with captions/subtitles enabled.")

    return {
        "title": data.get("title", "Unknown"),
        "author": data.get("channel", {}).get("title", "Unknown"),
        "transcript": transcript_text.strip(),
        "duration": data.get("durationFormatted", "Unknown"),
        "thumbnail": data.get("thumbnail", ""),
        "view_count": data.get("viewCountInt", 0),
        "like_count": data.get("likeCountInt", 0),
    }


# ============================================================================
# LangGraph Workflow
# ============================================================================


def analysis_node(state: SummarizerState, analysis_model: str) -> dict:
    """Generate analysis from transcript."""
    llm = ChatOpenAI(
        model=analysis_model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
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


def quality_node(state: SummarizerState, quality_model: str) -> dict:
    """Assess quality of analysis."""
    llm = ChatOpenAI(
        model=quality_model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    ).with_structured_output(Quality)

    system_prompt = "Evaluate the analysis and provide each aspect a rating (Pass/Refine/Fail) with reasons."
    if state.target_language:
        system_prompt += f" Verify that the analysis is in {state.target_language}."

    analysis_json = state.analysis.model_dump_json() if state.analysis else "No analysis"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{state.transcript}\n\nAnalysis:\n{analysis_json}"),
    ]

    quality: Quality = llm.invoke(messages)

    return {
        "quality": quality,
        "is_complete": quality.is_acceptable,
    }


def should_continue(state: SummarizerState) -> str:
    """Determine next step in workflow."""
    if state.is_complete:
        return END

    if state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        return "analysis"

    return END


def create_summarizer_graph(analysis_model: str, quality_model: str) -> StateGraph:
    """Create the summarization workflow graph."""
    builder = StateGraph(SummarizerState)

    builder.add_node("analysis", lambda s: analysis_node(s, analysis_model))
    builder.add_node("quality", lambda s: quality_node(s, quality_model))

    builder.add_edge(START, "analysis")
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


def summarize_transcript(
    transcript: str,
    analysis_model: str,
    quality_model: str,
    target_language: str = "en",
) -> tuple[Analysis, Optional[Quality], int]:
    """Summarize transcript with quality self-checking.

    Returns:
        tuple: (analysis, quality, iteration_count)
    """
    graph = create_summarizer_graph(analysis_model, quality_model)

    initial_state = SummarizerState(
        transcript=transcript,
        target_language=target_language,
    )

    result = graph.invoke(initial_state.model_dump())
    final_state = SummarizerState.model_validate(result)

    return final_state.analysis, final_state.quality, final_state.iteration_count


# ============================================================================
# Main Actor Logic
# ============================================================================


async def main() -> None:
    """Main entry point for the YouTube Summarizer Actor."""
    async with Actor:
        # Charge for Actor start
        await Actor.charge("actor-start")

        # Get input
        actor_input = await Actor.get_input()

        youtube_urls = actor_input.get("youtubeUrls", [])
        analysis_model = actor_input.get("analysisModel", "x-ai/grok-4.1-fast")
        quality_model = actor_input.get("qualityModel", "x-ai/grok-4.1-fast")
        target_language = actor_input.get("targetLanguage", "en")
        max_videos = actor_input.get("maxVideos", 10)

        if not youtube_urls:
            raise ValueError('Missing "youtubeUrls" attribute in input!')

        # Get API keys from environment
        scrapecreators_api_key = os.getenv("SCRAPECREATORS_API_KEY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if not scrapecreators_api_key:
            raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Set OpenRouter API key for langchain
        os.environ["OPENAI_API_KEY"] = openrouter_api_key

        Actor.log.info(f"Processing {len(youtube_urls)} YouTube URLs with {analysis_model}")

        # Process each video
        processed_count = 0
        for idx, url in enumerate(youtube_urls[:max_videos], 1):
            try:
                Actor.log.info(f"[{idx}/{len(youtube_urls)}] Processing: {url}")
                start_time = time.time()

                # Scrape video
                Actor.log.info("Scraping video metadata and transcript...")
                video_data = scrape_youtube_video(url, scrapecreators_api_key)
                Actor.log.info(f'Video: {video_data["title"]} by {video_data["author"]}')

                # Analyze transcript
                Actor.log.info("Analyzing transcript with LangGraph workflow...")
                analysis, quality, iterations = summarize_transcript(
                    transcript=video_data["transcript"],
                    analysis_model=analysis_model,
                    quality_model=quality_model,
                    target_language=target_language,
                )

                processing_time = time.time() - start_time
                quality_score = quality.percentage_score if quality else None

                Actor.log.info(f"Analysis complete: {iterations} iterations, quality {quality_score}%, {processing_time:.1f}s")

                # Extract video ID from URL
                import re

                video_id_match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", url)
                video_id = video_id_match.group(1) if video_id_match else "unknown"

                # Format chapters as markdown
                chapters_md = ""
                for chapter in analysis.chapters:
                    chapters_md += f"\n## {chapter.header}\n\n"
                    chapters_md += f"{chapter.summary}\n\n"
                    for point in chapter.key_points:
                        chapters_md += f"- {point}\n"

                # Format the complete markdown output
                formatted_output = f"""**URL:** {url}

**Title:** {analysis.title}

**Thumbnail:** {video_data["thumbnail"]}

**Channel:** {video_data["author"]}


# Summary

{analysis.summary}

# Key Takeaways

{chr(10).join(f'- {takeaway}' for takeaway in analysis.takeaways)}

# Video Chapters
{chapters_md}
# Keywords

{', '.join(analysis.keywords)}"""

                # Push to dataset
                await Actor.push_data(
                    {
                        "video_id": video_id,
                        "url": url,
                        "title": analysis.title,
                        "thumbnail": video_data["thumbnail"],
                        "channel": video_data["author"],
                        "duration": video_data["duration"],
                        "summary": analysis.summary,
                        "takeaways": analysis.takeaways,
                        "chapters": [c.model_dump() for c in analysis.chapters],
                        "keywords": analysis.keywords,
                        "formatted_output": formatted_output,
                        "iteration_count": iterations,
                        "processing_time": f"{processing_time:.1f}s",
                        "processed_at": datetime.now().isoformat(),
                    }
                )

                processed_count += 1
                await Actor.charge("video-analyzed")

            except Exception as e:
                Actor.log.error(f"Failed to process {url}: {e}")
                continue

        Actor.log.info(f"Successfully processed {processed_count}/{len(youtube_urls)} videos")
