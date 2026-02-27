# Scripts Module Codemap

## Scope

- Target module: `scripts/`
- Related modules: `mcp_server.py` and `youtube_summarizer/settings.py` (deployed runtime consumes these env vars/secrets).

## What Module Is For

- `scripts/` contains operational deployment automation, currently centered on Cloud Run environment/secret reconciliation and service deployment.

## High-signal locations

- `scripts/deploy_cloud_run.sh -> cleanup`, `read_env_file_value`, `load_service_env_cache`, `read_service_env_value`
- `scripts/deploy_cloud_run.sh -> ensure_secret_with_value`, `resolve_value`
- `scripts/deploy_cloud_run.sh -> gcloud run deploy` invocation block

## Repository snapshot

- Filesystem: 1 directory, 2 files (`1` shell, `1` Markdown).
- AST/script metrics: no Python/TS symbols in this module; one module codemap file.
- Syntax edges: deployment script has no import graph; behavior is command-driven.

## Symbol Inventory

- Shell functions:
  - `scripts/deploy_cloud_run.sh -> cleanup`
  - `scripts/deploy_cloud_run.sh -> read_env_file_value`
  - `scripts/deploy_cloud_run.sh -> load_service_env_cache`
  - `scripts/deploy_cloud_run.sh -> load_service_status_url`
  - `scripts/deploy_cloud_run.sh -> read_service_env_value`
  - `scripts/deploy_cloud_run.sh -> ensure_secret_with_value`
  - `scripts/deploy_cloud_run.sh -> resolve_value`
- Runtime variables and wiring pivots:
  - Service metadata cache (`SERVICE_ENV_CACHE`, status URL probes).
  - Secret map and env map split between `--set-secrets` and `--set-env-vars`.

## Syntax Relationships

- `scripts/deploy_cloud_run.sh -> gcloud run services describe`: reads current URL/env/service account.
- `scripts/deploy_cloud_run.sh -> gcloud secrets create/versions add`: idempotent secret hydration.
- `scripts/deploy_cloud_run.sh -> gcloud run deploy --set-env-vars/--set-secrets`: applies runtime wiring.
- `scripts/deploy_cloud_run.sh -> OAuth callback printout`: aligns with FastMCP Google OAuth expectations in `mcp_server.py`.

## Key takeaways per location

- `resolve_value`: precedence is explicit env var -> local `.env` -> existing Cloud Run env/secret references.
- `ensure_secret_with_value`: secret creation and version updates are idempotent per deploy run.
- Deploy command block keeps runtime env vars and secret mounts distinct (`--set-env-vars` vs `--set-secrets`).

## Project-specific conventions and rationale

- OAuth settings are treated as required in the Cloud Run path (`MCP_AUTH_MODE=google_oauth`).
- Script synchronizes `MCP_SERVER_BASE_URL` with regional/service URL and prints the exact callback URI.
- At least one LLM key and one transcript key must exist before deploy proceeds.
- Sensitive values stay in Secret Manager; do not move secrets into plain env values.
- Cloud Storage bucket for FastMCP OAuth state must remain writable by the runtime service account.
