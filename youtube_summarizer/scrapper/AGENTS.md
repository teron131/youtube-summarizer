# Scrapper Submodule Codemap

## Scope

- Target module: `youtube_summarizer/scrapper/`
- Related modules: `routes/scrape.py`, `routes/summarize.py`, and `mcp_server.py` call this module for transcript extraction.

## What Module Is For

- `scrapper/` integrates transcript providers and exposes normalized transcript retrieval for API and MCP summarization flows.

## High-signal locations

- `youtube_summarizer/scrapper/scrapper.py -> has_transcript_provider_key`, `_fetch_scrape_creators`, `_fetch_supadata`, `scrape_youtube`, `get_transcript`, `extract_transcript_text`
- `youtube_summarizer/scrapper/__init__.py -> exported symbols`
- `youtube_summarizer/scrapper/scrape_creators.py -> scrape_youtube` (single-provider helper)
- `youtube_summarizer/scrapper/supadata.py -> fetch_supadata_transcript` (single-provider helper)

## Repository snapshot

- Filesystem: 1 directory, 5 files (`4` Python, `1` Markdown).
- AST metrics: `__all__` exports = 1, Python import edges = 15.
- Dependency posture: module is provider-integration boundary; no independent entrypoint.

## Symbol Inventory

- Data models:
  - `scrapper.py -> Channel`, `TranscriptSegment`, `YouTubeScrapperResult`
  - `scrape_creators.py -> Channel`, `TranscriptSegment`, `YouTubeScrapperResult`
- Runtime provider flow:
  - `scrapper.py -> has_transcript_provider_key`, `_fetch_scrape_creators`, `_fetch_supadata`, `scrape_youtube`, `get_transcript`, `extract_transcript_text`
  - `supadata.py -> get_supadata_api_key`, `fetch_supadata_transcript`
  - `scrape_creators.py -> scrape_youtube`
- Public exports:
  - `__init__.py -> Channel`, `TranscriptSegment`, `YouTubeScrapperResult`, `extract_transcript_text`, `has_transcript_provider_key`, `scrape_youtube`

## Syntax Relationships

- `routes/scrape.py -> scrape_video -> youtube_summarizer.scrapper.extract_transcript_text`
- `routes/summarize.py -> _summarize_with_provider -> youtube_summarizer.scrapper.extract_transcript_text`
- `mcp_server.py -> scrape/summarize -> youtube_summarizer.scrapper.extract_transcript_text`
- `scrapper.py -> settings.get_settings`
- `scrapper.py -> utils.clean_youtube_url`, `utils.is_youtube_url`, `utils.clean_text`

## Key takeaways per location

- `scrapper.py -> scrape_youtube`: primary runtime path; provider fallback and validation live here.
- `scrapper.py -> YouTubeScrapperResult.parsed_transcript`: transcript normalization is centralized and reused by callers.
- `scrapper.py -> has_transcript_provider_key`: key-presence gate for HTTP and MCP endpoints.
- `__init__.py`: exports primary API; downstream imports should prefer package exports over deep file imports.
- `scrape_creators.py` and `supadata.py`: narrower provider helpers retained for compatibility and focused testing.

## Project-specific conventions and rationale

- Preserve provider order in `scrapper.py`:
  - Scrape Creators
  - Supadata
- Treat 401/403 and non-success HTTP responses as soft provider failures during fallback.
- Return normalized full transcript text; timestamp chunk detail is not part of API/MCP contracts.
- Missing both transcript keys is a configuration error, not a recoverable runtime warning.
- URL validation and cleanup happen before provider requests.
