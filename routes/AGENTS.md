# Routes Module Codemap

## Scope

- Target module: `routes/`
- Related modules: `youtube_summarizer/` (domain logic), `app.py` (FastAPI assembly), `mcp_server.py` (parallel MCP surface).

## What Module Is For

- `routes/` defines the FastAPI HTTP surface for scrape/summarize/health, validates request shape, resolves provider/language choices, and maps internal errors to HTTP responses.

## High-signal locations

- `routes/__init__.py -> api_router`
- `routes/summarize.py -> summarize`, `stream_summarize`, `_resolve_provider`, `_summarize_with_provider`
- `routes/scrape.py -> scrape_video`
- `routes/health.py -> root`, `health_check`, `get_configuration`
- `routes/schema.py -> SummarizeRequest`, `SummarizeResponse`, `ConfigurationResponse`
- `routes/errors.py -> create_http_error`, `handle_exception`
- `routes/helpers.py -> get_processing_time`, `run_async_task`

## Repository snapshot

- Filesystem: 1 directory, 8 files (`7` Python, `1` Markdown).
- AST metrics: Python `def` = 8, Python `async def` = 8, Python import edges = 41.
- Entrypoint posture: no standalone entrypoint inside module; mounted via `app.py`.

## Symbol Inventory

- Functions (handlers/orchestration):
  - `routes/summarize.py -> _require_any_llm_config`, `_validate_summary_request`, `_resolve_provider`, `_summarize_with_provider`, `_build_metadata`, `summarize`, `stream_summarize`
  - `routes/scrape.py -> scrape_video`
  - `routes/health.py -> root`, `health_check`, `get_configuration`
  - `routes/helpers.py -> run_async_task`, `get_processing_time`
  - `routes/errors.py -> create_http_error`, `handle_exception`
- Classes (contracts):
  - `routes/schema.py -> BaseResponse`, `YouTubeRequest`, `SummarizeRequest`, `ScrapeResponse`, `SummarizeResponse`, `ConfigurationResponse`
- Key module constants:
  - `routes/summarize.py -> AVAILABLE_PROVIDERS`, `SUPPORTED_LANGUAGES`, `DEFAULT_TARGET_LANGUAGE`, `LLM_CONFIG_ERROR`

## Syntax Relationships

- `app.py -> routes.api_router -> include_router(health/scrape/summarize)`
- `routes/summarize.py -> youtube_summarizer.summarizer_gemini.summarize_video_async`
- `routes/summarize.py -> youtube_summarizer.scrapper.extract_transcript_text -> youtube_summarizer.summarizer_openrouter.summarize_video_async`
- `routes/scrape.py -> youtube_summarizer.scrapper.extract_transcript_text`
- `routes/scrape.py -> youtube_summarizer.scrapper.has_transcript_provider_key`
- `routes/health.py -> youtube_summarizer.settings.get_settings().to_public_config`
- `routes/summarize.py -> routes.helpers.run_async_task/get_processing_time` and `routes.errors.handle_exception`

## Key takeaways per location

- `routes/__init__.py -> api_router`: single aggregation point for route registration; keep new route modules registered here.
- `routes/summarize.py -> _resolve_provider`: provider routing rule is deterministic (`gemini` preferred in `auto`, then `openrouter`).
- `routes/summarize.py -> _summarize_with_provider`: Gemini summarizes from URL directly; OpenRouter summarizes from extracted transcript.
- `routes/summarize.py -> stream_summarize`: SSE response emits JSON events (`status` then `complete` or `error`).
- `routes/scrape.py -> scrape_video`: transcript provider keys are required before extraction work.
- `routes/health.py -> get_configuration`: publishes safe runtime config via `settings.to_public_config()`.
- `routes/schema.py`: endpoint contracts are centralized; update models before changing handler payloads.
- `routes/errors.py -> handle_exception`: coarse error taxonomy (`quota` to 429, invalid input patterns to 400, fallback to 500).

## Project-specific conventions and rationale

- Keep URL-only behavior for scrape and summarize endpoints; transcript text stays internal.
- Preserve provider fallback semantics:
  - `provider=auto` resolves by available keys.
  - Explicit provider request fails loudly if its key is missing.
- Keep `target_language` constrained to `auto|en|zh` to match prompt/schema assumptions.
- Include lightweight metadata in responses (`processing_time`, optional token/cost stats).
- Do not leak secrets/config internals in response payloads; expose only `to_public_config()`.
