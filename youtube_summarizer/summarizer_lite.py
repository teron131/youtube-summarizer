"""YouTube video transcript summarization using LangChain ReAct agent with structured output."""

from collections.abc import Callable
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .llm_harness import ChatOpenRouter, filter_content, tag_content, untag_content
from .prompts import get_garbage_filter_prompt, get_langchain_summary_prompt
from .schemas import GarbageIdentification, Summary
from .scrapper import extract_transcript_text
from .utils import is_youtube_url

load_dotenv()

OPENROUTER_SUMMARY_MODEL = os.getenv("OPENROUTER_SUMMARY_MODEL", "x-ai/grok-4.1-fast")
OPENROUTER_FILTER_MODEL = os.getenv("OPENROUTER_FILTER_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025")


@tool
def scrape_youtube(youtube_url: str) -> str:
    """Scrape a YouTube video and return the transcript."""
    return extract_transcript_text(youtube_url)


@wrap_tool_call
def garbage_filter_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    result = handler(request)

    if request.tool_call["name"] == "scrape_youtube" and result.status != "error":
        transcript = result.content
        if isinstance(transcript, str) and transcript.strip():
            tagged_transcript = tag_content(transcript)
            llm = ChatOpenRouter(
                model=OPENROUTER_FILTER_MODEL,
                temperature=0,
            ).with_structured_output(GarbageIdentification)
            messages = [
                SystemMessage(content=get_garbage_filter_prompt()),
                HumanMessage(content=tagged_transcript),
            ]
            garbage: GarbageIdentification = llm.invoke(messages)

            if garbage.garbage_ranges:
                filtered_transcript = filter_content(tagged_transcript, garbage.garbage_ranges)
                result.content = untag_content(filtered_transcript)

    return result


def create_summarizer_agent(target_language: str | None = None):
    llm = ChatOpenRouter(
        model=OPENROUTER_SUMMARY_MODEL,
        temperature=0,
        reasoning_effort="medium",
    )

    agent = create_agent(
        model=llm,
        tools=[scrape_youtube],
        system_prompt=get_langchain_summary_prompt(target_language=target_language),
        middleware=[garbage_filter_middleware],
        response_format=ToolStrategy(Summary),
    )
    return agent


def summarize_video(
    transcript_or_url: str,
    target_language: str | None = None,
) -> Summary:
    agent = create_summarizer_agent(target_language=target_language)

    prompt = f"Summarize this YouTube video: {transcript_or_url}" if is_youtube_url(transcript_or_url) else f"Transcript:\n{transcript_or_url}"
    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    structured_response = response.get("structured_response")
    if structured_response is None:
        raise ValueError("Agent did not return structured response")

    return structured_response
