# YouTube Summarizer Backend API

![YouTube Summarizer UI](ui.png)

**Frontend:** [YouTube Summarizer UI](https://github.com/teron131/youtube-summarizer-ui)

A Python backend API for YouTube video analysis. It focuses on transcript-first processing with provider-based extraction and provider-routed summarization.

> **Status:** Active refactor toward a cleaner provider-mode pipeline.

## 🌟 Key Features

- **🎯 Unified API Layer**: Backend endpoints for scraping, summarizing, and streaming progress updates.
- **🧾 Provider-Based Transcript Flow**: Uses Scrape Creators / Supadata for transcript and metadata retrieval.
- **✂️ Simplified Transcript Shape**: Normalized transcript text only (no timestamp-level transcript in API output).
- **🤖 Single Summarization Path**: One production-focused summarization flow with configurable LLM provider.
- **🔌 Flexible LLM Routing**: Supports Gemini/OpenRouter model backends via environment configuration.
- **⚡ API-First Design**: FastAPI endpoints designed for UI integration and streaming UX.

## 🗺️ Planned Workflow (Better-Youtube Style)

The following workflow reflects the target architecture plan for this repo.

```mermaid
graph TD
  R{Auto or keys fallback}

  G[[Gemini API]]

  T{Transcript API}
  SC[[Scrape Creators API]]
  SD[[Supadata API]]

  TXT[Transcript / Metadata]
  OR[[OpenRouter API]]
  S[Summarization]

  R --> T
  T --> SC --> TXT
  T --> SD --> TXT

  R --> G
  G --> S
  TXT --> OR --> S

  classDef api fill:#E3F2FD,stroke:#1565C0,color:#0D47A1;
  classDef ui fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20;
  classDef option fill:#FFF3E0,stroke:#EF6C00,color:#E65100;

  class G,OR,SC,SD api;
  class S ui;
  class R,T option;
```

## 📌 Plan Notes

- The backend remains transcript-first: transcript APIs feed the single summarization path.
- Transcript payloads are normalized to text segments/blocks; timestamp-level transcript detail is out of scope.
- Runtime behavior is now simplified to provider selection (Gemini/OpenRouter) instead of summary mode selection.
- Router behavior is explicit: it resolves `auto` first, then falls back to whichever provider key is available.
- `/summarize` and `/stream-summarize` accept `provider: auto | gemini | openrouter` (default `auto`).
- `/summarize` and `/stream-summarize` accept YouTube `url` only; transcript text remains an internal processing step.

## 🚀 Setup & Development

### Prerequisites

- Python 3.12+
- API Keys: `SCRAPECREATORS_API_KEY` and/or `SUPADATA_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` (optional)

### 1. Installation

```bash
# Recommended: Use UV
uv sync
uv pip install -e .

# Alternative: Use pip
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file (see `.env_example`):

```env
SCRAPECREATORS_API_KEY=...
SUPADATA_API_KEY=... # Optional transcript provider
TRANSCRIPT_PROVIDER_PREFERENCE=scrapecreators # scrapecreators | supadata
GEMINI_API_KEY=...
OPENROUTER_API_KEY=... # Optional
GEMINI_SUMMARY_MODEL=gemini-3-flash-preview
OPENROUTER_SUMMARY_MODEL=x-ai/grok-4.1-fast
OPENROUTER_REASONING_EFFORT=medium
TARGET_LANGUAGE=en
PORT=8080
```

Model selection is environment-controlled. API callers choose only `provider` (`auto|gemini|openrouter`) and cannot pass model IDs.
Language control is also constrained: API callers can use `target_language` as `auto`, `en`, or `zh` only. `zh` output is always Traditional Chinese (繁體中文).

### 3. Execution

```bash
# Development
python app.py

# Production
./start.sh
```

## 🎯 API Reference

| Endpoint            | Method | Description                                                        |
| ------------------- | ------ | ------------------------------------------------------------------ |
| `/scrape`           | `POST` | Extract video metadata and normalized transcript text via provider |
| `/summarize`        | `POST` | Generate AI summary (`provider` route with `auto` fallback)        |
| `/stream-summarize` | `POST` | Real-time streaming updates of summarization progress              |
| `/health`           | `GET`  | System status and API configuration check                          |

_Interactive docs available at `/api/docs` or `/api/redoc`._

## 🧰 MCP Server (FastMCP)

This repo includes a standalone FastMCP server at `mcp_server.py` (no FastAPI route dependency) that exposes tools:

- `health`
- `config`
- `scrape`
- `summarize`

Run over stdio:

```bash
uv run python mcp_server.py
```

Or with FastMCP CLI:

```bash
uv run fastmcp run mcp_server.py
```

Run over HTTP transport:

```bash
MCP_TRANSPORT=http MCP_HOST=0.0.0.0 MCP_PORT=8000 uv run python mcp_server.py
```
