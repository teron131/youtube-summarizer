"""OpenRouter/Gemini LLM client initialization and configuration."""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def _is_openrouter(model: str) -> bool:
    """Check if model is OpenRouter format (PROVIDER/MODEL)."""
    return "/" in model and len(model.split("/")) == 2


def _is_gemini(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.lower().startswith("gemini")


def _get_config(model: str, api_key: str | None = None) -> tuple[str | None, str]:
    """Get API key and base URL based on model type."""
    if _is_openrouter(model):
        return api_key or os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1"

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key, "https://generativelanguage.googleapis.com/v1beta/openai/"


def ChatOpenRouter(
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    web_search: bool = False,
    web_search_engine: Literal["native", "exa"] | None = None,
    web_search_max_results: int = 5,
    pdf_engine: Literal["mistral-ocr", "pdf-text", "native"] | None = None,
    cached_content: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize OpenRouter or Gemini models."""
    if not (_is_openrouter(model) or _is_gemini(model)):
        raise ValueError(f"Invalid model: {model}")

    api_key, base_url = _get_config(model)
    extra_body = kwargs.pop("extra_body", {}) or {}

    if _is_openrouter(model):
        if provider_sort and "provider" not in extra_body:
            extra_body["provider"] = {"sort": provider_sort}

        plugins = extra_body.get("plugins", [])

        if pdf_engine:
            plugins.append({"id": "file-parser", "pdf": {"engine": pdf_engine}})

        if web_search:
            web_plugin: dict[str, str | int] = {"id": "web"}
            if web_search_engine:
                web_plugin["engine"] = web_search_engine
            if web_search_max_results != 5:
                web_plugin["max_results"] = web_search_max_results
            plugins.append(web_plugin)

        if plugins:
            extra_body["plugins"] = plugins
    elif cached_content:
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


def OpenRouterEmbeddings(
    model: str = "openai/text-embedding-3-small",
    **kwargs,
) -> OpenAIEmbeddings:
    """Initialize an OpenRouter embedding model."""
    if not _is_openrouter(model):
        raise ValueError(f"Invalid OpenRouter model format: {model}. Expected PROVIDER/MODEL")

    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=kwargs.pop("check_embedding_ctx_length", False),
        **kwargs,
    )
