# llm_harness Submodule Codemap

## Scope

- Target module: `youtube_summarizer/llm_harness/`
- Related modules: `youtube_summarizer/summarizer_openrouter.py` (primary consumer), `youtube_summarizer/settings.py` (model/key defaults).

## What Module Is For

- `llm_harness/` encapsulates provider-aware LangChain model initialization and text-tag filtering helpers used by summarization paths.

## High-signal locations

- `youtube_summarizer/llm_harness/openrouter.py -> _is_openrouter`, `_is_gemini`, `_get_config`, `ChatOpenRouter`, `OpenRouterEmbeddings`
- `youtube_summarizer/llm_harness/fast_copy.py -> TagRange`, `tag_content`, `untag_content`, `filter_content`
- `youtube_summarizer/llm_harness/__init__.py -> exported facade`

## Repository snapshot

- Filesystem: 1 directory, 4 files (`3` Python, `1` Markdown).
- AST metrics: `__all__` exports = 1, Python import edges = 8.
- Relationship posture: submodule is consumed by summarizer code; no independent service entrypoint.

## Symbol Inventory

- Model routing and client factories:
  - `openrouter.py -> _is_openrouter`, `_is_gemini`, `_get_config`, `ChatOpenRouter`, `OpenRouterEmbeddings`
- Text span utilities:
  - `fast_copy.py -> TagRange`, `tag_content`, `untag_content`, `filter_content`
- Public exports:
  - `__init__.py -> ChatOpenRouter`, `TagRange`, `tag_content`, `untag_content`, `filter_content`

## Syntax Relationships

- `summarizer_openrouter.py -> ChatOpenRouter(...).with_structured_output(Summary)`
- `summarizer_openrouter.py -> get_settings -> openrouter_summary_model/openrouter_reasoning_effort`
- `openrouter.py -> settings.get_settings`
- `__init__.py -> re-export ChatOpenRouter and fast_copy helpers`

## Key takeaways per location

- `openrouter.py -> ChatOpenRouter`: one constructor handles both OpenRouter and Gemini-compatible endpoints.
- `openrouter.py -> _get_config`: API key and base URL are derived from model string, not caller branching.
- `openrouter.py`: OpenRouter-specific settings are attached through guarded `extra_body` plugin/provider options.
- `openrouter.py -> OpenRouterEmbeddings`: enforces OpenRouter model format (`provider/model`) with shared base URL.
- `fast_copy.py`: utility layer for tagged spans and range filtering used by backup summarizer flows.

## Project-specific conventions and rationale

- Keep model string parsing strict:
  - OpenRouter: `PROVIDER/MODEL`
  - Gemini: `gemini*`
- Preserve model names/defaults from `settings.py` unless references are updated end-to-end.
- Resolve secrets through `get_settings()`; avoid passing secret strings down call stacks.
- Keep `ChatOpenRouter` interface stable because summarizer modules depend on structured output behavior.
