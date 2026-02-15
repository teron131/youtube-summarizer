# Scripts Module Codemap

## Scope

- Operational scripts for deploy/runtime configuration, currently centered on Cloud Run deployment.

## High-signal locations

- `scripts/deploy_cloud_run.sh -> resolve_value`, `read_service_env_value`, `ensure_secret_with_value`
- `scripts/deploy_cloud_run.sh -> gcloud run deploy` invocation

## Key takeaways per location

- `resolve_value`: configuration precedence is explicit env var -> local `.env` -> current Cloud Run service/env+secret references.
- `ensure_secret_with_value`: secret creation and version updates are idempotent per deployment run.
- deploy command block: runtime env vars and secret mounts are split (`--set-env-vars` vs `--set-secrets`).

## Project-specific conventions and rationale

- OAuth settings are treated as required for Cloud Run deploy in this script (`MCP_AUTH_MODE=google_oauth` path).
- Deploy script synchronizes `MCP_SERVER_BASE_URL` with regional/service URL and prints exact callback URI.
- At least one LLM key and one transcript key must exist before deploy proceeds.
- Sensitive values should remain in Secret Manager; never shift them into plain env values in script changes.
- Cloud Storage bucket is used for FastMCP OAuth state volume and should stay writable by runtime service account.

## Syntax relationship highlights (ast-grep-first)

- `scripts/deploy_cloud_run.sh -> gcloud run services describe -> read current URL/env/service account`
- `scripts/deploy_cloud_run.sh -> gcloud secrets create/versions add -> secret hydration`
- `scripts/deploy_cloud_run.sh -> gcloud run deploy --set-env-vars/--set-secrets -> runtime wiring`

## General approach (not rigid checklist)

- Keep script idempotent and safe for repeated deploys.
- Prefer helper functions for env/secret resolution over inline one-off parsing.
- Guard every required value before deploy starts, with explicit operator-facing error text.
- Preserve backward compatibility with `.env` usage for local deploy workflows.

## Validation commands

```bash
bash -n scripts/deploy_cloud_run.sh
/Users/teron/Projects/Agents-Config/.factory/hooks/formatter.sh
```
