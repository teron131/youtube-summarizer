"""YouTube video summarization using Google Gemini multimodal API."""

import logging

from google import genai
from google.genai import types

from .prompts import get_gemini_summary_prompt
from .schemas import Summary
from .settings import get_settings

logger = logging.getLogger(__name__)

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


def _extract_usage_metadata(response) -> dict[str, int | float] | None:
    settings = get_settings()
    metadata: dict[str, int | float] | None = None
    usage = getattr(response, "usage_metadata", None)
    if usage and hasattr(usage, "prompt_token_count") and hasattr(usage, "total_token_count"):
        tokens_input = int(usage.prompt_token_count)
        tokens_total = int(usage.total_token_count)
        tokens_output = max(0, tokens_total - tokens_input)
        cost = _calculate_cost(settings.gemini_summary_model, tokens_input, tokens_total)
        metadata = {
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_total": tokens_total,
            "cost": cost,
        }
        logger.info(
            "Gemini usage input=%s total=%s est_cost_usd=%.6f",
            tokens_input,
            tokens_total,
            cost,
        )
    return metadata


def analyze_video_url(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
) -> tuple[Summary | None, dict[str, int | float] | None]:
    settings = get_settings()
    api_key = api_key or settings.google_api_key or settings.gemini_api_key
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY")

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": settings.llm_timeout_milliseconds},
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_summary_model,
            contents=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=get_gemini_summary_prompt(target_language=target_language)),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=settings.gemini_thinking_level),
                response_mime_type="application/json",
                response_schema=Summary,
            ),
        )

        if not response.text:
            logger.warning("Empty response from Gemini API")
            return None, None

        summary = Summary.model_validate_json(response.text)
        return summary, _extract_usage_metadata(response)

    except Exception as e:
        logger.exception("Failed to analyze video: %s", e)
        return None, None


def summarize_video(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
) -> tuple[Summary | None, dict[str, int | float] | None]:
    return analyze_video_url(
        video_url,
        target_language=target_language,
        api_key=api_key,
    )


async def analyze_video_url_async(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
) -> tuple[Summary | None, dict[str, int | float] | None]:
    settings = get_settings()
    api_key = api_key or settings.google_api_key or settings.gemini_api_key
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY")

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": settings.llm_timeout_milliseconds},
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_summary_model,
            contents=[
                types.Part(file_data=types.FileData(file_uri=video_url)),
                types.Part(text=get_gemini_summary_prompt(target_language=target_language)),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=settings.gemini_thinking_level),
                response_mime_type="application/json",
                response_schema=Summary,
            ),
        )

        if not response.text:
            logger.warning("Empty response from Gemini API")
            return None, None

        summary = Summary.model_validate_json(response.text)
        return summary, _extract_usage_metadata(response)
    except Exception as e:
        logger.exception("Failed to analyze video: %s", e)
        return None, None


async def summarize_video_async(
    video_url: str,
    *,
    target_language: str = "auto",
    api_key: str | None = None,
) -> tuple[Summary | None, dict[str, int | float] | None]:
    return await analyze_video_url_async(
        video_url,
        target_language=target_language,
        api_key=api_key,
    )
