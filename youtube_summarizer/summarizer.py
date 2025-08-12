"""
Content Summarization Service
-----------------------------

This module provides functions for processing transcribed text to generate
formatted subtitles and AI-powered summaries using the Gemini API.
"""

import os

from google.genai import Client, types


def quick_summary(text: str) -> str:
    """
    Generates a summary of the provided text using the Gemini API.
    The prompt is designed to produce a concise summary with key facts,
    maintaining the original language of the text.
    """
    try:
        client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=os.getenv("LLM"),
            contents=f"Summarize with list out of the key facts mentioned. Follow the language of the text.\n\n{text}",
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text
    except Exception as e:
        return f"Summary generation failed: {str(e)}"


def simple_format_subtitle(subtitle: str) -> str:
    """
    Performs basic formatting on a raw transcript.

    - Capitalizes the first letter of each line.
    - Ensures each line ends with a punctuation mark.
    """
    lines = subtitle.split("\n")
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Capitalize the first letter if it's a lowercase character
            if line[0].islower():
                line = line[0].upper() + line[1:]
            # Add a period if the line doesn't end with punctuation
            if not line.endswith((".", "!", "?")):
                line += "."
            formatted_lines.append(line)
    return "\n".join(formatted_lines)
