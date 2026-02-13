"""OpenRouter transcript summarization for deployment runtime."""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from .llm_harness import ChatOpenRouter
from .prompts import get_langchain_summary_prompt
from .schemas import Summary

load_dotenv()

OPENROUTER_SUMMARY_MODEL = os.getenv("OPENROUTER_SUMMARY_MODEL", "x-ai/grok-4.1-fast")
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "medium")


def summarize_video(
    transcript: str,
    target_language: str | None = None,
) -> Summary:
    clean_transcript = transcript.strip()
    if not clean_transcript:
        raise ValueError("Transcript cannot be empty")

    llm = ChatOpenRouter(
        model=OPENROUTER_SUMMARY_MODEL,
        temperature=0,
        reasoning_effort=OPENROUTER_REASONING_EFFORT,
    ).with_structured_output(Summary)

    messages = [
        SystemMessage(content=get_langchain_summary_prompt(target_language=target_language)),
        HumanMessage(content=f"Transcript:\n{clean_transcript}"),
    ]
    return llm.invoke(messages)
