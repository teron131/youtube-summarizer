# youtube_summarizer Module Codemap

## Scope

- Target module: `youtube_summarizer/`
- Related modules: `routes/` (HTTP orchestration), `mcp_server.py` (FastMCP orchestration), `app.py` (server assembly).

## What Module Is For

- `youtube_summarizer/` is the core domain package for transcript acquisition, LLM summarization, shared schemas/prompts, and runtime settings consumed by both HTTP and MCP entrypoints.

## High-signal locations

- `youtube_summarizer/__init__.py -> package exports`
- `youtube_summarizer/settings.py -> AppSettings`, `get_settings`
- `youtube_summarizer/schemas.py -> Summary`, `Chapter`
- `youtube_summarizer/prompts.py -> _build_context_block`, `get_gemini_summary_prompt`, `get_langchain_summary_prompt`
- `youtube_summarizer/summarizer_gemini.py -> summarize_video_async`, `analyze_video_url_async`
- `youtube_summarizer/summarizer_openrouter.py -> summarize_video_async`
- `youtube_summarizer/scrapper/scrapper.py -> scrape_youtube`, `extract_transcript_text`
- `youtube_summarizer/llm_harness/openrouter.py -> ChatOpenRouter`
- `youtube_summarizer/utils.py -> clean_youtube_url`, `is_youtube_url`, `s2hk`

## Repository snapshot

- Filesystem: 3 directories, 17 files (`14` Python, `3` Markdown).
- AST metrics: `__all__` exports = 3, Python import edges = 51, module codemap files = 3.
- Operational posture: no direct entrypoint file in package; consumed by `routes/` and `mcp_server.py`.

## Symbol Inventory

- Settings and schema:
  - `settings.py -> AppSettings`, `_clean_optional`, `get_settings`
  - `schemas.py -> Chapter`, `Summary`
- Prompt and summarization pipeline:
  - `prompts.py -> _build_context_block`, `get_gemini_summary_prompt`, `get_langchain_summary_prompt`
  - `summarizer_gemini.py -> _calculate_cost`, `_extract_usage_metadata`, `analyze_video_url`, `summarize_video`, `analyze_video_url_async`, `summarize_video_async`
  - `summarizer_openrouter.py -> summarize_video`, `summarize_video_async`
- Utilities:
  - `utils.py -> clean_text`, `clean_youtube_url`, `is_youtube_url`, `extract_video_id`, `s2hk`, `whisper_result_to_txt`, `safe_truncate`, `serialize_nested`
- Public export facade:
  - `__init__.py -> Summary`, `extract_transcript_text`, `has_transcript_provider_key`, `summarize_video_gemini`, `summarize_video_openrouter`, `get_settings`

## Syntax Relationships

- `mcp_server.py -> youtube_summarizer.__init__ exports -> summarize_video_gemini/summarize_video_openrouter`
- `routes/summarize.py -> youtube_summarizer.summarizer_gemini.summarize_video_async`
- `routes/summarize.py -> youtube_summarizer.scrapper.extract_transcript_text -> youtube_summarizer.summarizer_openrouter.summarize_video_async`
- `routes/scrape.py -> youtube_summarizer.scrapper.extract_transcript_text`
- `summarizer_openrouter.py -> llm_harness.ChatOpenRouter -> langchain_openai.ChatOpenAI`
- `scrapper/scrapper.py -> settings.get_settings` and transcript provider endpoint config

## Key takeaways per location

- `youtube_summarizer/__init__.py`: stable import surface consumed by `mcp_server.py`.
- `settings.py -> get_settings`: single cached settings object; optional secrets normalized by stripping blanks.
- `schemas.py`: canonical summary contract (overview + chronological chapters).
- `prompts.py`: language and anti-hallucination constraints are encoded in prompt builders.
- `summarizer_gemini.py`: URL-native multimodal summarize path with usage/cost metadata extraction.
- `summarizer_openrouter.py`: transcript-first summarize path with structured output validation.
- `scrapper/scrapper.py`: transcript provider fallback order is Scrape Creators first, then Supadata.
- `llm_harness/openrouter.py`: model-id format controls backend routing (OpenRouter vs Gemini OpenAI-compatible endpoint).
- `utils.py`: URL normalization and Traditional Chinese conversion helpers act as shared invariants.

## Project-specific conventions and rationale

- Transcript-first architecture remains default for OpenRouter and scrape flows.
- Gemini path accepts video URL directly and does not require transcript provider success.
- Runtime provider routing is key-based; no mode matrix or manual multipath orchestration.
- `Summary` output stays schema-validated to keep API and MCP responses stable.
- Chinese normalization uses `s2hk` via schema validators, enforcing output consistency at model level.
- Public diagnostics must exclude secrets; use `AppSettings.to_public_config()` for outward-facing config.
- Preserve existing model names unless every consuming reference is updated together.
