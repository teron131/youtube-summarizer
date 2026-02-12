"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv()


def _is_openrouter(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def _is_gemini(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.lower().startswith("gemini")


def _get_config(model: str, api_key: str | None = None) -> tuple[str, str]:
    """Get API key and base URL based on model type."""
    if _is_openrouter(model):
        return api_key or os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1"

    # Default to Gemini/Google (via OpenAI compatibility endpoint)
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key, "https://generativelanguage.googleapis.com/v1beta/openai/"


def ChatOpenRouter(
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    pdf_engine: Literal["mistral-ocr", "pdf-text", "native"] | None = None,
    cached_content: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize OpenRouter or Gemini model.

    Args:
        model: PROVIDER/MODEL for OpenRouter, or gemini-* for Gemini
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: "minimal", "low", "medium", "high"
        provider_sort: OpenRouter routing - "throughput", "price", "latency"
        pdf_engine: "mistral-ocr" (scanned), "pdf-text" (structured), "native"
        cached_content: Gemini cached content ID
        **kwargs: Additional arguments passed to ChatOpenAI
    """
    if not (_is_openrouter(model) or _is_gemini(model)):
        raise ValueError(f"Invalid model: {model}")

    api_key, base_url = _get_config(model)
    extra_body = kwargs.pop("extra_body", {}) or {}

    if _is_openrouter(model):
        if provider_sort and "provider" not in extra_body:
            extra_body["provider"] = {"sort": provider_sort}
        if pdf_engine:
            plugins = extra_body.get("plugins", [])
            plugins.append({"id": "file-parser", "pdf": {"engine": pdf_engine}})
            extra_body["plugins"] = plugins

    elif _is_gemini(model) and cached_content:
        extra_body.setdefault("google", {})["cached_content"] = cached_content

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body or None,
        **kwargs,
    )
