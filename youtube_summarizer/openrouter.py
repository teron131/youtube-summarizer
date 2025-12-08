"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def _is_openrouter_model(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def _is_gemini_model(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.lower().startswith("gemini")


def _get_openrouter_config(api_key: Optional[str] = None) -> tuple[str, str]:
    """Get API key and base URL for OpenRouter models."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    return api_key, base_url


def _get_gemini_config(api_key: Optional[str] = None) -> tuple[str, str]:
    """Get API key and base URL for Gemini models."""
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    return api_key, base_url


def ChatOpenRouter(
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    pdf_engine: Optional[Literal["mistral-ocr", "pdf-text", "native"]] = None,
    cached_content: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize an OpenRouter model with sensible defaults.

    Args:
        model: Model identifier (PROVIDER/MODEL)
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: Level of reasoning effort for reasoning models.
            Can be "minimal", "low", "medium", or "high".
        provider_sort: Routing preference for OpenRouter (default: "throughput")
        pdf_engine: PDF processing engine for file attachments. Options:
            - "mistral-ocr": Best for scanned documents ($2 per 1,000 pages)
            - "pdf-text": Best for well-structured PDFs (Free)
            - "native": Use model's native file processing (if available)
            If None, OpenRouter will auto-select based on model capabilities.
        cached_content: Gemini cached content ID (e.g., "cachedContents/0000aaaa1111bbbb2222cccc3333dddd4444eeee")
            Only used for Gemini models. Pass the full cached content resource name.
        **kwargs: Additional config (e.g. max_tokens, extra_body, etc.)

    Note: Some models (e.g., google/gemini-2.5-pro) may not support PDFs through OpenRouter in the same way as others. If you encounter "invalid_prompt" errors with PDFs, try a different model.
    """
    if not (_is_openrouter_model(model) or _is_gemini_model(model)):
        raise ValueError(f"Invalid model: {model}. Use 'PROVIDER/MODEL' for OpenRouter or a Gemini model identifier.")

    extra_body: dict[str, Any] | None = None
    if _is_openrouter_model(model):
        api_key, base_url = _get_openrouter_config()
        extra_body = _build_openrouter_extra_body(
            base_extra_body=kwargs.pop("extra_body", {}) or {},
            provider_sort=provider_sort,
            pdf_engine=pdf_engine,
        )
    elif _is_gemini_model(model):
        api_key, base_url = _get_gemini_config()
        extra_body = _build_gemini_extra_body(
            base_extra_body=kwargs.pop("extra_body", {}) or {},
            cached_content=cached_content,
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body or None,
        **kwargs,
    )


def _build_openrouter_extra_body(
    *,
    base_extra_body: dict[str, Any],
    provider_sort: Optional[Literal["throughput", "price", "latency"]],
    pdf_engine: Optional[Literal["mistral-ocr", "pdf-text", "native"]],
) -> dict[str, Any]:
    """Build OpenRouter extra_body config."""
    extra = {**base_extra_body}

    if provider_sort and "provider" not in extra:
        extra["provider"] = {"sort": provider_sort}

    if pdf_engine:
        plugins = [*extra.get("plugins", []), {"id": "file-parser", "pdf": {"engine": pdf_engine}}]
        extra["plugins"] = plugins

    return extra


def _build_gemini_extra_body(
    *,
    base_extra_body: dict[str, Any],
    cached_content: Optional[str],
) -> dict[str, Any] | None:
    """Build Gemini extra_body config with cached content support."""
    extra = {**base_extra_body}

    if cached_content:
        # Merge with existing google config if present
        google_config = extra.get("google", {})
        google_config["cached_content"] = cached_content
        extra["google"] = google_config

    return extra
