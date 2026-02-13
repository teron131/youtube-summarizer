"""OpenRouter transcript summarization for deployment runtime."""

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_harness import ChatOpenRouter
from .prompts import get_langchain_summary_prompt
from .schemas import Summary
from .settings import get_settings


def summarize_video(
    transcript: str,
    target_language: str | None = None,
) -> Summary:
    settings = get_settings()
    clean_transcript = transcript.strip()
    if not clean_transcript:
        raise ValueError("Transcript cannot be empty")

    llm = ChatOpenRouter(
        model=settings.openrouter_summary_model,
        temperature=0,
        reasoning_effort=settings.openrouter_reasoning_effort,
        timeout=settings.llm_timeout_seconds,
    ).with_structured_output(Summary)

    messages = [
        SystemMessage(content=get_langchain_summary_prompt(target_language=target_language)),
        HumanMessage(content=f"Transcript:\n{clean_transcript}"),
    ]
    return llm.invoke(messages)


async def summarize_video_async(
    transcript: str,
    target_language: str | None = None,
) -> Summary:
    settings = get_settings()
    clean_transcript = transcript.strip()
    if not clean_transcript:
        raise ValueError("Transcript cannot be empty")

    llm = ChatOpenRouter(
        model=settings.openrouter_summary_model,
        temperature=0,
        reasoning_effort=settings.openrouter_reasoning_effort,
        timeout=settings.llm_timeout_seconds,
    ).with_structured_output(Summary)

    messages = [
        SystemMessage(content=get_langchain_summary_prompt(target_language=target_language)),
        HumanMessage(content=f"Transcript:\n{clean_transcript}"),
    ]
    return await llm.ainvoke(messages)
