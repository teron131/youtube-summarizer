"""YouTube video summarization using Google Gemini multimodal API."""

import logging
import os

from google import genai
from google.genai import types

from .prompts import get_gemini_summary_prompt
from .schemas import Summary

logger = logging.getLogger(__name__)

GEMINI_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-3-flash-preview")
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "medium")

USD_PER_M_TOKENS_BY_MODEL = {
    "gemini-3-flash-preview": {"input": 0.5, "output": 3},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
}


def _calculate_cost(
    model: str,
    prompt_tokens: int,
    total_tokens: int,
) -> float:
    pricing = USD_PER_M_TOKENS_BY_MODEL.get(model)
    if not pricing:
        return 0.0

    output_tokens = max(0, total_tokens - prompt_tokens)
    return (prompt_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def analyze_video_url(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
    timeout: int = 600,
) -> Summary | None:
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY")

    client = genai.Client(api_key=api_key, http_options={"timeout": timeout})

    try:
        response = client.models.generate_content(
            model=GEMINI_SUMMARY_MODEL,
            contents=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=get_gemini_summary_prompt(target_language=target_language)),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
                response_mime_type="application/json",
                response_schema=Summary,
            ),
        )

        if not response.text:
            logger.warning("Empty response from Gemini API")
            return None

        summary = Summary.model_validate_json(response.text)

        usage = getattr(response, "usage_metadata", None)
        if usage and hasattr(usage, "prompt_token_count") and hasattr(usage, "total_token_count"):
            cost = _calculate_cost(GEMINI_SUMMARY_MODEL, usage.prompt_token_count, usage.total_token_count)
            logger.info(
                "Gemini usage input=%s total=%s est_cost_usd=%.6f",
                usage.prompt_token_count,
                usage.total_token_count,
                cost,
            )

        return summary

    except Exception as e:
        logger.exception("Failed to analyze video: %s", e)
        return None


def summarize_video(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
) -> Summary | None:
    return analyze_video_url(
        video_url,
        target_language=target_language,
        api_key=api_key,
    )
