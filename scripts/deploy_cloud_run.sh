#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-prime-hologram-487308-a3}"
REGION="${REGION:-asia-east1}"
SERVICE_NAME="${SERVICE_NAME:-youtube-summarizer-mcp}"
ENV_FILE="${ENV_FILE:-.env}"
SERVICE_ENV_CACHE_JSON=""
SERVICE_STATUS_URL=""

cleanup() {
  if [[ -n "${SERVICE_ENV_CACHE_JSON}" && -f "${SERVICE_ENV_CACHE_JSON}" ]]; then
    rm -f "${SERVICE_ENV_CACHE_JSON}"
  fi
}

trap cleanup EXIT

read_env_file_value() {
  local key="$1"
  local file_path="$2"

  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi

  local line
  line="$(grep -E "^${key}=" "${file_path}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 0
  fi

  local value
  value="${line#*=}"
  value="${value%$'\r'}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"

  printf "%s" "${value}"
}

load_service_env_cache() {
  if [[ -n "${SERVICE_ENV_CACHE_JSON}" ]]; then
    return 0
  fi

  local cache_file
  cache_file="$(mktemp)"

  if gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format=json >"${cache_file}" 2>/dev/null; then
    SERVICE_ENV_CACHE_JSON="${cache_file}"
    return 0
  fi

  rm -f "${cache_file}"
}

load_service_status_url() {
  if [[ -n "${SERVICE_STATUS_URL}" ]]; then
    return 0
  fi

  SERVICE_STATUS_URL="$(
    gcloud run services describe "${SERVICE_NAME}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --format='value(status.url)' 2>/dev/null || true
  )"
}

read_service_env_value() {
  local key="$1"

  load_service_env_cache
  if [[ -z "${SERVICE_ENV_CACHE_JSON}" || ! -f "${SERVICE_ENV_CACHE_JSON}" ]]; then
    return 0
  fi

  local result
  result="$(
    uv run python - "${key}" "${SERVICE_ENV_CACHE_JSON}" <<'PY'
import json
import sys

key = sys.argv[1]
path = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

env_list = (
    data.get("spec", {})
    .get("template", {})
    .get("spec", {})
    .get("containers", [{}])[0]
    .get("env", [])
)

for item in env_list:
    if item.get("name") != key:
        continue
    if "value" in item:
        print(f"VALUE::{item['value']}", end="")
        break
    secret_ref = item.get("valueFrom", {}).get("secretKeyRef", {})
    secret_name = secret_ref.get("name") or secret_ref.get("secret")
    if secret_name:
        print(f"SECRET::{secret_name}", end="")
        break
PY
  )"

  if [[ -z "${result}" ]]; then
    return 0
  fi

  if [[ "${result}" == VALUE::* ]]; then
    printf "%s" "${result#VALUE::}"
    return 0
  fi

  if [[ "${result}" == SECRET::* ]]; then
    local secret_name
    secret_name="${result#SECRET::}"
    gcloud secrets versions access latest \
      --project "${PROJECT_ID}" \
      --secret "${secret_name}" 2>/dev/null || true
  fi
}

ensure_secret_with_value() {
  local secret_name="$1"
  local secret_value="$2"

  if [[ -z "${secret_value}" ]]; then
    return 0
  fi

  if ! gcloud secrets describe "${secret_name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud secrets create "${secret_name}" \
      --project "${PROJECT_ID}" \
      --replication-policy="automatic" >/dev/null
  fi

  printf "%s" "${secret_value}" | gcloud secrets versions add "${secret_name}" \
    --project "${PROJECT_ID}" \
    --data-file=- >/dev/null
}

resolve_value() {
  local key="$1"
  local explicit_value="$2"

  if [[ -n "${explicit_value}" ]]; then
    printf "%s" "${explicit_value}"
    return 0
  fi

  local env_file_value
  env_file_value="$(read_env_file_value "${key}" "${ENV_FILE}")"
  if [[ -n "${env_file_value}" ]]; then
    printf "%s" "${env_file_value}"
    return 0
  fi

  read_service_env_value "${key}"
}

load_service_status_url
MCP_SERVER_BASE_URL="${MCP_SERVER_BASE_URL:-${SERVICE_STATUS_URL}}"
MCP_GOOGLE_CLIENT_ID="$(resolve_value MCP_GOOGLE_CLIENT_ID "${MCP_GOOGLE_CLIENT_ID:-}")"
MCP_GOOGLE_CLIENT_SECRET="$(resolve_value MCP_GOOGLE_CLIENT_SECRET "${MCP_GOOGLE_CLIENT_SECRET:-}")"

GEMINI_API_KEY="$(resolve_value GEMINI_API_KEY "${GEMINI_API_KEY:-}")"
OPENROUTER_API_KEY="$(resolve_value OPENROUTER_API_KEY "${OPENROUTER_API_KEY:-}")"
SCRAPECREATORS_API_KEY="$(resolve_value SCRAPECREATORS_API_KEY "${SCRAPECREATORS_API_KEY:-}")"
SUPADATA_API_KEY="$(resolve_value SUPADATA_API_KEY "${SUPADATA_API_KEY:-}")"

if [[ -z "${MCP_GOOGLE_CLIENT_ID}" || -z "${MCP_GOOGLE_CLIENT_SECRET}" ]]; then
  echo "Missing OAuth env vars. Set MCP_GOOGLE_CLIENT_ID and MCP_GOOGLE_CLIENT_SECRET (or place them in ${ENV_FILE})."
  exit 1
fi

if [[ -z "${MCP_SERVER_BASE_URL}" ]]; then
  echo "Missing MCP_SERVER_BASE_URL and unable to read existing service URL."
  echo "Set MCP_SERVER_BASE_URL explicitly (env var or ${ENV_FILE})."
  exit 1
fi

if [[ -z "${GEMINI_API_KEY}" && -z "${OPENROUTER_API_KEY}" ]]; then
  echo "Missing LLM key. Set GEMINI_API_KEY or OPENROUTER_API_KEY (env var or ${ENV_FILE})."
  exit 1
fi

if [[ -z "${SCRAPECREATORS_API_KEY}" && -z "${SUPADATA_API_KEY}" ]]; then
  echo "Missing transcript key. Set SCRAPECREATORS_API_KEY or SUPADATA_API_KEY (env var or ${ENV_FILE})."
  exit 1
fi

gcloud services enable secretmanager.googleapis.com \
  --project "${PROJECT_ID}" >/dev/null

ensure_secret_with_value "MCP_GOOGLE_CLIENT_SECRET" "${MCP_GOOGLE_CLIENT_SECRET}"
ensure_secret_with_value "GEMINI_API_KEY" "${GEMINI_API_KEY}"
ensure_secret_with_value "OPENROUTER_API_KEY" "${OPENROUTER_API_KEY}"
ensure_secret_with_value "SCRAPECREATORS_API_KEY" "${SCRAPECREATORS_API_KEY}"
ensure_secret_with_value "SUPADATA_API_KEY" "${SUPADATA_API_KEY}"

ENV_VARS=(
  "MCP_TRANSPORT=http"
  "MCP_HOST=0.0.0.0"
  "DEFAULT_TARGET_LANGUAGE=auto"
  "MCP_AUTH_MODE=google_oauth"
  "MCP_SERVER_BASE_URL=${MCP_SERVER_BASE_URL}"
  "MCP_GOOGLE_CLIENT_ID=${MCP_GOOGLE_CLIENT_ID}"
)

ENV_VARS_CSV="$(IFS=,; echo "${ENV_VARS[*]}")"

SECRET_VARS=(
  "MCP_GOOGLE_CLIENT_SECRET=MCP_GOOGLE_CLIENT_SECRET:latest"
)

if [[ -n "${GEMINI_API_KEY}" ]]; then
  SECRET_VARS+=("GEMINI_API_KEY=GEMINI_API_KEY:latest")
fi
if [[ -n "${OPENROUTER_API_KEY}" ]]; then
  SECRET_VARS+=("OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest")
fi
if [[ -n "${SCRAPECREATORS_API_KEY}" ]]; then
  SECRET_VARS+=("SCRAPECREATORS_API_KEY=SCRAPECREATORS_API_KEY:latest")
fi
if [[ -n "${SUPADATA_API_KEY}" ]]; then
  SECRET_VARS+=("SUPADATA_API_KEY=SUPADATA_API_KEY:latest")
fi

SECRET_VARS_CSV="$(IFS=,; echo "${SECRET_VARS[*]}")"

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 4 \
  --timeout 180 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "${ENV_VARS_CSV}" \
  --set-secrets "${SECRET_VARS_CSV}" \
  --command=web

FINAL_SERVICE_URL="$(
  gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format='value(status.url)' 2>/dev/null || true
)"

if [[ -n "${FINAL_SERVICE_URL}" && "${FINAL_SERVICE_URL}" != "${MCP_SERVER_BASE_URL}" ]]; then
  gcloud run services update "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --update-env-vars "MCP_SERVER_BASE_URL=${FINAL_SERVICE_URL}" >/dev/null
  MCP_SERVER_BASE_URL="${FINAL_SERVICE_URL}"
fi

echo
echo "Deploy complete."
echo "Service URL: ${FINAL_SERVICE_URL:-unknown}"
echo "OAuth Redirect URI (Google client): ${MCP_SERVER_BASE_URL%/}/auth/callback"
