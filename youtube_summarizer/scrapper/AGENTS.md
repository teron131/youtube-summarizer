# Scrapper Submodule Codemap

## Scope

- Transcript provider integration and normalized transcript extraction.
- Exposes async transcript text retrieval used by API and MCP summarization flows.

## High-signal locations

- `youtube_summarizer/scrapper/scrapper.py -> scrape_youtube`, `_fetch_scrape_creators`, `_fetch_supadata`, `extract_transcript_text`
- `youtube_summarizer/scrapper/__init__.py -> exported symbols`
- `youtube_summarizer/scrapper/scrape_creators.py -> scrape_youtube` (single-provider legacy helper)
- `youtube_summarizer/scrapper/supadata.py -> fetch_supadata_transcript` (single-provider legacy helper)

## Key takeaways per location

- `scrapper.py -> scrape_youtube`: primary runtime path; provider fallback and validation live here.
- `scrapper.py -> YouTubeScrapperResult.parsed_transcript`: transcript normalization is centralized and reused by callers.
- `scrapper.py -> has_transcript_provider_key`: key-presence gate used by HTTP and MCP endpoints.
- `__init__.py`: exports primary API; downstream imports should prefer package exports over deep file imports.
- `scrape_creators.py` and `supadata.py`: narrower provider-specific helpers retained for compatibility/testing.

## Project-specific conventions and rationale

- Preserve provider order in `scrapper.py`:
  1. Scrape Creators
  2. Supadata
- Treat 401/403 and non-success HTTP responses as soft provider failures during fallback.
- Return normalized full text transcript for consumers; timestamp chunk detail is not returned by API/MCP contracts.
- Missing both transcript keys is a configuration error, not a recoverable runtime warning.
- URL validation and cleanup happen before provider requests.

## Syntax relationship highlights (ast-grep-first)

- `routes/scrape.py -> scrape_video -> youtube_summarizer.scrapper.extract_transcript_text`
- `routes/summarize.py -> _summarize_with_provider -> youtube_summarizer.scrapper.extract_transcript_text`
- `mcp_server.py -> scrape/summarize -> youtube_summarizer.scrapper.extract_transcript_text`
- `youtube_summarizer/scrapper/scrapper.py -> settings.get_settings`
- `youtube_summarizer/scrapper/scrapper.py -> utils.clean_youtube_url`, `utils.is_youtube_url`, `utils.clean_text`

## General approach (not rigid checklist)

- Add new transcript providers in `scrapper.py` first, with explicit fallback position.
- Keep provider adapters tolerant to response-shape drift (`extra="ignore"` models, guarded parsing).
- Prefer returning `None` from provider fetch helpers and letting `scrape_youtube` decide terminal error behavior.
- Keep public error messages concise and key-actionable for ops.

## Validation commands

```bash
uv run ruff check youtube_summarizer/scrapper
uv run ruff format youtube_summarizer/scrapper
/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh
```
