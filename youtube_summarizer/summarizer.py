"""
Content Summarization Service
-----------------------------

This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using the Gemini API.
"""

import os

from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel, Field

from .utils import is_youtube_url

load_dotenv()


class Chapter(BaseModel):
    header: str = Field(description="A descriptive title for the chapter")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")


class Analysis(BaseModel):
    title: str = Field(description="The main title or topic of the video content")
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience")
    overall_summary: str = Field(description="A comprehensive summary synthesizing all chapters, facts, and themes")


def summarize_video(url_or_caption: str) -> Analysis:
    """
    Summarize the text using the Gemini. Streaming seems to be less buggy with long videos.
    """
    client = Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"timeout": 600000},
    )

    response = client.models.generate_content_stream(
        model="models/gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=url_or_caption)) if is_youtube_url(url_or_caption) else types.Part(text=url_or_caption),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction="Analyze the video/transcript according to the schema and follow the original language.",
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    result_parts = [chunk.text for chunk in response if chunk.text is not None]
    final_result = "".join(result_parts)
    final_result = Analysis.model_validate_json(final_result)  # Convert to Pydantic model

    return final_result
