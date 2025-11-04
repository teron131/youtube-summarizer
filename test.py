"""
Minimalistic test script to get transcript using Scrape Creators API
and generate analysis using LangChain agent (not LangGraph).

Environment Variables Required:
    - SCRAPECREATORS_API_KEY: Your Scrape Creators API key
    - OPENROUTER_API_KEY: Your OpenRouter API key (or GEMINI_API_KEY for Gemini)

Usage:
    uv run python test.py [youtube_url] [model_name]

Examples:
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "google/gemini-2.5-flash"
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "google/gemini-2.5-pro"
"""

import json
import os
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()


class Chapter(BaseModel):
    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")


class Analysis(BaseModel):
    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience")
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned")
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="The exact keywords in the analysis worthy of highlighting")
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


def get_transcript_from_scrape_creators(youtube_url: str) -> str:
    """Get transcript from Scrape Creators API."""
    api_key = os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")

    print(f"🔗 Fetching transcript from Scrape Creators API...")
    start_time = time.time()

    url = f"https://api.scrapecreators.com/v1/youtube/video?url={youtube_url}&get_transcript=true"
    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    result = response.json()
    transcript = result.get("transcript_only_text", "")

    elapsed = time.time() - start_time
    print(f"✅ Transcript fetched in {elapsed:.2f}s ({len(transcript)} characters)")

    if not transcript:
        raise ValueError("No transcript found in API response")

    return transcript


def create_llm(model: str):
    """Create LangChain LLM instance based on model format."""
    if "/" in model:
        # OpenRouter format (e.g., "google/gemini-2.5-flash")
        return init_chat_model(
            model=model,
            model_provider="openai",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
    else:
        # Direct Gemini format (e.g., "gemini-2.5-flash")
        return init_chat_model(
            model=model,
            model_provider="google_genai",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )


def analyze_with_agent(transcript: str, model: str = "google/gemini-2.5-flash") -> Analysis:
    """Analyze transcript using LangChain agent (not LangGraph).

    Args:
        transcript: The video transcript text
        model: Model name (defaults to fast flash model)
    """
    print(f"🤖 Starting analysis with LangChain agent...")
    print(f"📝 Model: {model}")
    print(f"📝 Transcript length: {len(transcript)} characters")

    start_time = time.time()

    # Create LLM
    llm = create_llm(model)

    # Create agent with direct structured output (simpler and faster)
    # No tools needed - agent will use structured output directly
    agent = create_agent(
        model=llm,
        tools=[],  # No tools needed for structured output
        response_format=Analysis,
        system_prompt="""Create a comprehensive analysis that strictly follows the transcript content.

CORE REQUIREMENTS:
- ACCURACY: Every claim must be directly supported by the transcript
- LENGTH: Summary (150-400 words), Chapters (80-200 words each), Takeaways (3-8), Key Facts (3-6)
- TONE: Write in objective, article-like style (avoid "This video...", "The speaker...")
- AVOID META-DESCRIPTIVE LANGUAGE: Do not use phrases like "This chapter introduces", "This section covers", "This analysis explores", etc. Write direct, factual content only

CONTENT FILTERING:
- Remove all promotional content (speaker intros, calls-to-action, self-promotion)
- Keep only educational content about the strategy
- Correct obvious typos naturally

CHAPTER REQUIREMENTS:
Create 4-8 thematic chapters based on content structure and topic transitions.

QUALITY CHECKS:
- Content matches transcript exactly (no external additions)
- All promotional content removed (intros, calls-to-action, self-promotion)
- Typos corrected naturally, meaning preserved
- Length balanced: substantial but not overwhelming
- Keywords highly relevant and searchable

LENGTH GUIDELINES:
- Title: 2-15 words
- Summary: 150-400 words
- Chapters: 80-200 words each
- Takeaways: 3-8 items
- Key Facts: 3-6 items
- Keywords: Exactly 3""",
    )

    # Create analysis prompt
    prompt = f"""Analyze this video transcript and create a comprehensive analysis:

{transcript}

Extract all key information accurately from the transcript, structure it into thematic chapters, and identify the most important takeaways and facts."""

    try:
        print("🔄 Generating analysis with agent...")
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # Extract structured response
        if "structured_response" in result:
            analysis: Analysis = result["structured_response"]
        else:
            # Fallback: try to parse from messages
            last_message = result["messages"][-1].content
            # Try to extract JSON from the message
            import re

            json_match = re.search(r"\{.*\}", last_message, re.DOTALL)
            if json_match:
                analysis_dict = json.loads(json_match.group())
                analysis = Analysis(**analysis_dict)
            else:
                raise ValueError("Could not extract structured response from agent")

        elapsed = time.time() - start_time
        print(f"✅ Analysis completed in {elapsed:.2f}s")
        print(f"📊 Generated {len(analysis.chapters)} chapters")

        return analysis

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Main test function."""
    # Default test URL
    test_url = "https://youtu.be/pmdiKAE_GLs"
    model = "x-ai/grok-4-fast"  # Fast default model

    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]

    print("=" * 80)
    print("MINIMALISTIC TEST: Scrape Creators API + LangChain Agent")
    print("=" * 80)
    print(f"📹 Video URL: {test_url}")
    print(f"🤖 Model: {model}\n")

    try:
        # Step 1: Get transcript
        transcript = get_transcript_from_scrape_creators(test_url)

        # Step 2: Analyze with LangChain agent
        analysis = analyze_with_agent(transcript, model=model)

        # Step 3: Display results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print(f"\n📌 Title: {analysis.title}")
        print(f"\n📝 Summary: {analysis.summary[:200]}...")
        print(f"\n🎯 Takeaways ({len(analysis.takeaways)}):")
        for i, takeaway in enumerate(analysis.takeaways, 1):
            print(f"  {i}. {takeaway}")
        print(f"\n📊 Key Facts ({len(analysis.key_facts)}):")
        for i, fact in enumerate(analysis.key_facts, 1):
            print(f"  {i}. {fact}")
        print(f"\n📚 Chapters ({len(analysis.chapters)}):")
        for i, chapter in enumerate(analysis.chapters, 1):
            print(f"  {i}. {chapter.header}")
            print(f"     {chapter.summary[:100]}...")
        print(f"\n🔑 Keywords: {', '.join(analysis.keywords)}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
