"""Archived prompts used by ReAct/LangGraph summarization flows."""


def get_garbage_filter_prompt() -> str:
    """Build the system prompt for garbage filtering."""
    return (
        "Identify transcript lines that are NOT part of the core content and should be removed.\n"
        "Focus on: sponsors/ads/promos, discount codes, affiliate links, subscribe/like/call to action blocks, filler intros/outros, housekeeping, and other irrelevant segments.\n"
        "The transcript contains line tags like [L1], [L2], etc.\n"
        "Return ONLY the line ranges to remove (garbage_ranges).\n"
        "If unsure about a segment, prefer excluding it."
    )


def get_quality_check_prompt(target_language: str | None = None) -> str:
    """Build the system prompt for quality assessment."""
    system_prompt = (
        "Evaluate the summary against the transcript.\n"
        "For each aspect in the response schema, return a rating (Fail/Refine/Pass) and a specific, actionable reason.\n"
        "Rules:\n"
        "- Be strict about transcript grounding\n"
        "- Treat any sponsor/promo/call to action content as a failure for no_garbage\n"
        "- Treat meta-language as a failure for meta_language_avoidance"
    )
    if target_language:
        system_prompt += f"\nVerify the output language matches: {target_language}"

    return system_prompt
