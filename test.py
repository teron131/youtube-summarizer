"""
Minimalistic test script to get transcript using Scrape Creators API
and generate analysis using deepagents (not LangGraph).

Installation:
    uv add deepagents

Environment Variables Required:
    - SCRAPECREATORS_API_KEY: Your Scrape Creators API key
    - OPENROUTER_API_KEY: Your OpenRouter API key (for structured output conversion)

Usage:
    uv run python test.py [youtube_url]

Example:
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
"""

import json
import os
import sys
import time
from typing import Optional

import requests
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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


def get_transcript_from_scrape_creators(youtube_url: str) -> str:
    """Get transcript from Scrape Creators API."""
    api_key = os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")

    print(f"🔗 Fetching transcript from Scrape Creators API for: {youtube_url}")
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


def analyze_with_deepagent(transcript: str, model: Optional[str] = None, use_structured_output: bool = True) -> Analysis:
    """Analyze transcript using deepagents.

    Args:
        transcript: The video transcript text
        model: Optional model name (defaults to deepagents default)
        use_structured_output: If True, use LangChain structured output after deepagents
    """
    print(f"🤖 Starting analysis with deepagents...")
    print(f"📝 Transcript length: {len(transcript)} characters")

    start_time = time.time()

    # Create analysis prompt
    analysis_prompt = f"""Analyze the following video transcript and create a comprehensive structured analysis.

OUTPUT REQUIREMENTS:
- Title: 2-15 words describing the main topic
- Summary: 150-400 words comprehensive summary
- Takeaways: 3-8 key insights
- Key Facts: 3-6 important facts, statistics, or data points
- Chapters: 4-8 thematic chapters (each with header, summary, and key_points)
- Keywords: Exactly 3 highly relevant keywords

CORE REQUIREMENTS:
- ACCURACY: Every claim must be directly supported by the transcript
- TONE: Write in objective, article-like style (avoid "This video...", "The speaker...")
- AVOID META-DESCRIPTIVE LANGUAGE: Do not use phrases like "This chapter introduces", "This section covers"
- CONTENT FILTERING: Remove all promotional content (speaker intros, calls-to-action, self-promotion)

TRANSCRIPT:
{transcript[:10000]}"""  # Limit transcript length for deepagents (can be adjusted)

    if len(transcript) > 10000:
        print(f"⚠️  Transcript truncated to 10000 characters for deepagents processing")

    # Create deep agent
    # Note: deepagents uses LangGraph under the hood but provides a simpler interface
    agent = create_deep_agent(
        system_prompt="""You are an expert content analyst. Your job is to analyze video transcripts and create comprehensive, structured summaries.

IMPORTANT: Write your analysis to a file called 'analysis.json' in valid JSON format matching this structure:
{
  "title": "string",
  "summary": "string",
  "takeaways": ["string"],
  "key_facts": ["string"],
  "chapters": [{"header": "string", "summary": "string", "key_points": ["string"]}],
  "keywords": ["string"]
}""",
    )

    # Run the agent
    print("🔄 Running deepagent analysis...")
    try:
        result = agent.invoke({"messages": [("user", analysis_prompt)]})

        # Extract the final message content
        if isinstance(result, dict) and "messages" in result:
            final_message = result["messages"][-1]
            analysis_text = final_message.content if hasattr(final_message, "content") else str(final_message)
        else:
            analysis_text = str(result)

        elapsed = time.time() - start_time
        print(f"✅ Deepagent analysis completed in {elapsed:.2f}s")
        print(f"📊 Analysis result length: {len(analysis_text)} characters")

        # Try to extract JSON from the response or use structured output
        if use_structured_output:
            print("🔄 Converting to structured output using LangChain...")
            return _convert_to_structured_output(analysis_text, transcript)
        else:
            # Try to parse JSON from the text response
            json_match = _extract_json_from_text(analysis_text)
            if json_match:
                return Analysis.model_validate(json_match)
            else:
                print("⚠️  Could not extract JSON, returning text analysis")
                return Analysis(title="Analysis Generated", summary=analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text, takeaways=["See full analysis above"], key_facts=["See full analysis above"], chapters=[], keywords=["analysis"])

    except Exception as e:
        print(f"❌ Deepagent error: {e}")
        # Fallback to direct LangChain structured output
        print("🔄 Falling back to direct LangChain structured output...")
        return _convert_to_structured_output(transcript, transcript)


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Try to extract JSON from text response."""
    # Look for JSON-like structures
    try:
        # Try to find JSON object in the text
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    return None


def _convert_to_structured_output(analysis_text: str, transcript: str) -> Analysis:
    """Convert text analysis to structured output using LangChain."""
    from langchain.prompts import ChatPromptTemplate

    # Use OpenRouter for fast models
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",  # Fast model
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(Analysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Convert the following analysis into the required structured format.
Ensure all fields are properly filled and the structure matches the Analysis schema exactly.""",
            ),
            (
                "human",
                """Analysis to convert:
{analysis_text}

Original transcript (for reference):
{transcript}""",
            ),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"analysis_text": analysis_text[:5000], "transcript": transcript[:2000]})  # Limit length  # Just for reference

    return result


def main():
    """Main test function."""
    # Test with a sample YouTube URL
    test_url = "https://youtu.be/pmdiKAE_GLs"  # Replace with your test URL

    if len(sys.argv) > 1:
        test_url = sys.argv[1]

    print("=" * 80)
    print("MINIMALISTIC TEST: Scrape Creators API + DeepAgent")
    print("=" * 80)
    print(f"📹 Video URL: {test_url}\n")

    try:
        # Step 1: Get transcript
        transcript = get_transcript_from_scrape_creators(test_url)

        # Step 2: Analyze with deepagent
        analysis = analyze_with_deepagent(transcript)

        print("\n✅ Test completed successfully!")
        print(f"📊 Generated analysis with {len(analysis.chapters)} chapters")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
