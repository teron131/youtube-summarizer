# Routes Module Codemap

## Scope

- This module defines the FastAPI HTTP surface for scrape/summarize/health.
- It validates request shape, resolves provider/language runtime choices, and converts internal errors to HTTP responses.

## High-signal locations

- `routes/__init__.py -> api_router`
- `routes/summarize.py -> summarize`, `stream_summarize`, `_resolve_provider`, `_summarize_with_provider`
- `routes/scrape.py -> scrape_video`
- `routes/health.py -> root`, `health_check`, `get_configuration`
- `routes/schema.py -> SummarizeRequest`, `SummarizeResponse`, `ConfigurationResponse`
- `routes/errors.py -> handle_exception`
- `routes/helpers.py -> get_processing_time`, `run_async_task`

## Key takeaways per location

- `routes/__init__.py -> api_router`: single aggregation point for route registration; keep new route modules registered here.
- `routes/summarize.py -> _resolve_provider`: provider routing rule is deterministic (`gemini` preferred in `auto`, then `openrouter`).
- `routes/summarize.py -> _summarize_with_provider`: only Gemini path summarizes from URL directly; OpenRouter path requires transcript extraction first.
- `routes/summarize.py -> stream_summarize`: SSE format is JSON events (`status` then `complete`/`error`).
- `routes/scrape.py -> scrape_video`: transcript provider keys are required before any extraction work.
- `routes/health.py -> get_configuration`: publishes safe runtime config via `settings.to_public_config()`.
- `routes/schema.py`: endpoint contracts are centralized here; update models before changing handler payloads.
- `routes/errors.py -> handle_exception`: coarse error taxonomy (`quota` -> 429, invalid input patterns -> 400, fallback -> 500).

## Project-specific conventions and rationale

- Keep URL-only behavior for scrape and summarize endpoints; transcript text is internal.
- Preserve provider fallback semantics:
  - `provider=auto` resolves by available keys.
  - Explicit provider request must fail loudly if its key is missing.
- `target_language` is constrained to `auto|en|zh` to match prompt/schema assumptions.
- API responses include lightweight metadata (`processing_time`, optional token/cost stats).
- Do not leak secrets/config internals in response payloads; use `to_public_config()` only.

## Syntax relationship highlights (ast-grep-first)

- `routes/summarize.py -> _summarize_with_provider -> youtube_summarizer.summarizer_gemini.summarize_video_async`
- `routes/summarize.py -> _summarize_with_provider -> youtube_summarizer.scrapper.extract_transcript_text -> youtube_summarizer.summarizer_openrouter.summarize_video_async`
- `routes/scrape.py -> scrape_video -> youtube_summarizer.scrapper.extract_transcript_text`
- `routes/scrape.py -> scrape_video -> youtube_summarizer.scrapper.has_transcript_provider_key`
- `routes/health.py -> get_configuration -> youtube_summarizer.settings.get_settings().to_public_config`

## General approach (not rigid checklist)

- Add/modify request and response models in `routes/schema.py` first.
- Keep handlers thin: validate input, call package-level APIs, map failures to HTTP errors.
- Prefer reusing `_resolve_provider`, `_build_metadata`, and `handle_exception` patterns over per-endpoint branching.
- Keep streaming and non-streaming summarize outputs semantically aligned.

## Validation commands

```bash
uv run ruff check .
uv run ruff format .
/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh
```
