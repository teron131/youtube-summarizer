"""
Content Summarization Service
-----------------------------

This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using the Gemini API.
"""

import os

from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel, Field

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


def summarize_video(caption: str) -> Analysis:
    """
    Summarize the text using the Gemini.
    """

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=types.Content(
            parts=[types.Part(text=caption)],
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    return response.parsed
