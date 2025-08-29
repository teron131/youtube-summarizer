"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using the Gemini API.
"""

import os

from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel, Field

from youtube_summarizer.utils import is_youtube_url

load_dotenv()


class Chapter(BaseModel):
    header: str = Field(description="A descriptive title for the chapter")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")


class Analysis(BaseModel):
    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience")
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned")
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="Keywords or topics mentioned in the video, max 3", max_length=3)


def summarize_video(url_or_transcript: str) -> Analysis:
    """
    Summarize the text using the Gemini. Streaming seems to be less buggy with long videos.
    """
    if is_youtube_url(url_or_transcript):
        print(f"🔗 Sending YouTube URL to Gemini: {url_or_transcript}")
    else:
        print(f"📝 Sending transcript text to Gemini: {len(url_or_transcript)} characters")
        print(f"📝 Text preview: {url_or_transcript[:200]}...")

    client = Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"timeout": 600000},
    )

    response = client.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=url_or_transcript)) if is_youtube_url(url_or_transcript) else types.Part(text=url_or_transcript),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction="""Analyze the video/transcript according to the schema.
The transcript provides starting timestamps for each sentence.
Add the timestamps [TIMESTAMP] at the end of the takeaways and key facts if available.
Consider the chapters (headers) if given but not necessary.
Ignore the promotional and meaningless content.
Follow the original language.""",
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    print(response.usage_metadata)
    return response.parsed
