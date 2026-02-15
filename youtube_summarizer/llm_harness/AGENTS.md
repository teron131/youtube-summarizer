# llm_harness Submodule Codemap

## Scope

- Provider-aware LangChain model initialization and content tagging helpers.
- Encapsulates OpenRouter/Gemini OpenAI-compatible client construction.

## High-signal locations

- `youtube_summarizer/llm_harness/openrouter.py -> ChatOpenRouter`, `_is_openrouter`, `_is_gemini`, `_get_config`
- `youtube_summarizer/llm_harness/openrouter.py -> OpenRouterEmbeddings`
- `youtube_summarizer/llm_harness/fast_copy.py -> tag_content`, `untag_content`, `filter_content`, `TagRange`
- `youtube_summarizer/llm_harness/__init__.py -> exported facade`

## Key takeaways per location

- `openrouter.py -> ChatOpenRouter`: one constructor handles both OpenRouter and Gemini-compatible endpoints.
- `openrouter.py -> _get_config`: API key/base URL derivation is model-driven, not caller-driven.
- `openrouter.py`: OpenRouter-specific features are attached through `extra_body` plugins/provider sort.
- `OpenRouterEmbeddings`: strictly OpenRouter model format (`provider/model`) with shared base URL.
- `fast_copy.py`: utility layer for tagged spans and filtered content used by backup summarizer pipelines.

## Project-specific conventions and rationale

- Keep model string parsing strict:
  - OpenRouter: `PROVIDER/MODEL`
  - Gemini: `gemini*`
- Preserve current model names and defaults from `settings.py` unless references are updated end-to-end.
- Route all secret resolution through `get_settings()`; avoid passing secrets around call stacks.
- Keep `ChatOpenRouter` interface stable because summarize modules depend on structured output behavior.

## Syntax relationship highlights (ast-grep-first)

- `youtube_summarizer/summarizer_openrouter.py -> ChatOpenRouter(...).with_structured_output(Summary)`
- `youtube_summarizer/summarizer_openrouter.py -> get_settings -> openrouter_summary_model/openrouter_reasoning_effort`
- `youtube_summarizer/llm_harness/openrouter.py -> settings.get_settings`
- `youtube_summarizer/llm_harness/__init__.py -> re-export ChatOpenRouter and fast_copy helpers`

## General approach (not rigid checklist)

- Extend provider behavior inside `ChatOpenRouter` instead of branching in summarize callers.
- Keep plugin additions additive and guarded; avoid mutating caller-provided objects unexpectedly.
- Validate new model-routing logic with both OpenRouter and Gemini model identifiers.
- Keep helper exports in `__init__.py` aligned with public usage.

## Validation commands

```bash
uv run ruff check youtube_summarizer/llm_harness
uv run ruff format youtube_summarizer/llm_harness
/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh
```
