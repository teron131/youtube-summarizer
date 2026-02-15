# youtube_summarizer Module Codemap

## Scope

- Core domain package for transcript acquisition, LLM summarization, shared schemas/prompts, and runtime settings.
- Both FastAPI routes and MCP tools call into this package.

## High-signal locations

- `youtube_summarizer/__init__.py -> package exports`
- `youtube_summarizer/settings.py -> AppSettings`, `get_settings`
- `youtube_summarizer/schemas.py -> Summary`, `Chapter`
- `youtube_summarizer/prompts.py -> get_gemini_summary_prompt`, `get_langchain_summary_prompt`
- `youtube_summarizer/summarizer_gemini.py -> summarize_video_async`
- `youtube_summarizer/summarizer_openrouter.py -> summarize_video_async`
- `youtube_summarizer/scrapper/scrapper.py -> scrape_youtube`, `extract_transcript_text`
- `youtube_summarizer/llm_harness/openrouter.py -> ChatOpenRouter`
- `youtube_summarizer/utils.py -> clean_youtube_url`, `is_youtube_url`, `s2hk`

## Key takeaways per location

- `youtube_summarizer/__init__.py`: defines the stable import surface consumed by `mcp_server.py`.
- `youtube_summarizer/settings.py -> get_settings`: single cached settings object; optional env secrets are normalized by stripping blanks.
- `youtube_summarizer/schemas.py`: canonical summary contract (overview + chronological chapters).
- `youtube_summarizer/prompts.py`: language requirement and anti-hallucination rules are encoded in prompt builders.
- `youtube_summarizer/summarizer_gemini.py`: URL-native multimodal summarize path with usage/cost metadata extraction.
- `youtube_summarizer/summarizer_openrouter.py`: transcript-first summarize path with structured output validation.
- `youtube_summarizer/scrapper/scrapper.py`: transcript provider fallback order is Scrape Creators first, then Supadata.
- `youtube_summarizer/llm_harness/openrouter.py`: model-id format controls backend routing (OpenRouter vs Gemini OpenAI-compatible endpoint).
- `youtube_summarizer/utils.py`: URL normalization and Traditional Chinese conversion helpers are shared invariants.

## Project-specific conventions and rationale

- Transcript-first architecture remains default for OpenRouter and scrape flows.
- Gemini path accepts video URL directly and does not require transcript provider success.
- Runtime provider routing is key-based; no mode matrix or manual multipath orchestration.
- `Summary` output is always schema-validated to keep API and MCP responses stable.
- Chinese normalization uses `s2hk` converter through schema validators, so output text consistency is enforced at model level.
- Public diagnostics must exclude secrets; use `AppSettings.to_public_config()` for outward-facing config.

## Syntax relationship highlights (ast-grep-first)

- `mcp_server.py -> youtube_summarizer.__init__ exports -> summarizer_gemini.summarize_video_async`
- `mcp_server.py -> youtube_summarizer.__init__ exports -> summarizer_openrouter.summarize_video_async`
- `routes/summarize.py -> youtube_summarizer.summarizer_gemini.summarize_video_async`
- `routes/summarize.py -> youtube_summarizer.scrapper.extract_transcript_text -> youtube_summarizer.summarizer_openrouter.summarize_video_async`
- `youtube_summarizer/summarizer_openrouter.py -> llm_harness.ChatOpenRouter -> langchain_openai.ChatOpenAI`
- `youtube_summarizer/scrapper/scrapper.py -> settings.get_settings` and provider endpoint URLs in settings

## General approach (not rigid checklist)

- Treat this package as the only place for provider logic and summary schema rules.
- Keep entrypoints (`routes`, `mcp_server.py`) orchestration-only; push shared behavior into package modules.
- When adding provider features, update:
  - settings (env + capability flags),
  - prompt/schema compatibility,
  - route and MCP fallback behavior.
- Preserve existing model names unless every consuming reference is updated together.

## Validation commands

```bash
uv run ruff check .
uv run ruff format .
/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh
```
