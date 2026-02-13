def _build_context_block(
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Helper to build the contextual information block."""
    metadata_parts = []
    if title:
        metadata_parts.append(f"Video Title: {title}")
    if description:
        metadata_parts.append(f"Video Description: {description}")

    if metadata_parts:
        return "\n# CONTEXTUAL INFORMATION:\n" + "\n".join(metadata_parts) + "\n"
    return ""


def get_gemini_summary_prompt(
    target_language: str = "auto",
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build the system prompt for video analysis (Gemini)."""
    # Language descriptions mapping
    lang_descriptions = {
        "auto": "Use the same language as the video, or English if the language is unclear",
        "en": "English (US)",
        "zh": "Traditional Chinese (繁體中文)",
    }

    # Determine language instruction
    normalized_language = target_language if target_language in {"auto", "en", "zh"} else "auto"
    lang_desc = lang_descriptions[normalized_language]
    instruction = lang_desc if normalized_language == "auto" else f"Write ALL output in {lang_desc}. Do not use English or any other language."

    language_instruction = f"- OUTPUT LANGUAGE (REQUIRED): {instruction}"

    metadata = _build_context_block(title, description)

    prompt_lines = [
        "Create a grounded, chronological summary.",
        metadata,
        language_instruction,
        "",
        "SOURCE: You are given the full video. Use BOTH spoken content and visuals (on-screen text/slides/charts/code/UI). Do not invent details that are not clearly supported by what you can see/hear.",
        "",
        "Return JSON only (no extra text) with:",
        "- overview: string",
        "- chapters: array of { title: string, description: string, start_time?: string, end_time?: string }",
        "(start_time/end_time are optional MM:SS; omit if unsure)",
        "",
        "Rules:",
        "- Chapters must be chronological and non-overlapping",
        "- Avoid meta-language (no 'this video...' framing)",
        "- Exclude sponsors/promos/calls to action entirely",
    ]

    return "\n".join(prompt_lines)


def get_langchain_summary_prompt(
    target_language: str | None = None,
    title: str | None = None,
    description: str | None = None,
) -> str:
    """Build the system prompt for transcript summarization (LangChain)."""
    metadata = _build_context_block(title, description)

    prompt_parts = [
        "Create a grounded, chronological summary of the transcript.",
        metadata,
        "Rules:",
        "- Ground every claim in the transcript; do not add unsupported details",
        "- Exclude sponsors/ads/promos/calls to action entirely",
        "- Avoid meta-language (no 'this video...', 'the speaker...', etc.)",
        "- Prefer concrete facts, names, numbers, and steps when present",
        "- Ensure output matches the provided response schema",
        "- Return JSON only with overview + chapters",
    ]

    if target_language:
        language_instruction = {
            "auto": "Use the same language as the transcript, or English if unclear",
            "en": "English (US)",
            "zh": "Traditional Chinese (繁體中文). Convert all Chinese text to Traditional Chinese.",
        }.get(target_language, "Use the same language as the transcript, or English if unclear")
        prompt_parts.append(f"\nOUTPUT LANGUAGE (REQUIRED): {language_instruction}")

    return "\n".join(prompt_parts)
