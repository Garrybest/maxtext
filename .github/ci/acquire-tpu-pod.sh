#!/usr/bin/env bash
set -euo pipefail

# acquire-tpu-pod.sh — atomically claim an idle pod from the CI TPU pool.
#
# Usage (sourced, not executed):
#   source .github/ci/acquire-tpu-pod.sh
#
# Required env vars:
#   JOB_NAME   — unique identifier for the calling workflow job
#   TASK_TYPE  — type of task being run (e.g. "smoke-test", "unit-tests")
#
# Exports on success:
#   POD        — name of the claimed pod

# ---------------------------------------------------------------------------
# 1. Validate required env vars
# ---------------------------------------------------------------------------
if [[ -z "${JOB_NAME:-}" ]]; then
  echo "ERROR: JOB_NAME env var is required but not set." >&2
  exit 1
fi

if [[ -z "${TASK_TYPE:-}" ]]; then
  echo "ERROR: TASK_TYPE env var is required but not set." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Ensure TPU pool deployment exists (auto-bootstrap)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Ensuring TPU pool infrastructure is up to date..."
kubectl apply -f "${SCRIPT_DIR}/tpu-pool-deployment.yaml"

# ---------------------------------------------------------------------------
# 3. Pool status summary
# ---------------------------------------------------------------------------
echo "=== CI TPU Pool Status ==="
kubectl get pods -l "pool=ci-tpu" \
  -o custom-columns=\
"NAME:.metadata.name,\
STATUS:.metadata.labels.status,\
CLAIMED-BY:.metadata.annotations.ci\.primatrix/claimed-by,\
CLAIMED-AT:.metadata.annotations.ci\.primatrix/claimed-at" \
  2>/dev/null || echo "(no pods found in pool)"
echo "=========================="

# ---------------------------------------------------------------------------
# 4. Poll loop
# ---------------------------------------------------------------------------
POLL_INTERVAL=15
MAX_WAIT=7200
ELAPSED=0

# Temp files for passing JSON between jq invocations (bash variables
# mangle escape sequences inside JSON strings, breaking re-parse).
POD_JSON_FILE=$(mktemp)
PATCHED_JSON_FILE=$(mktemp)
trap 'rm -f "${POD_JSON_FILE}" "${PATCHED_JSON_FILE}"' EXIT

while true; do
  # Query for an idle pod — write to file, not variable
  kubectl get pods -l "pool=ci-tpu,status=idle" \
    --field-selector=status.phase=Running -o json 2>/dev/null \
    | jq '.items[0] // empty' > "${POD_JSON_FILE}"

  # Check if we got a valid pod (file is non-empty and not "null")
  if [[ -s "${POD_JSON_FILE}" ]] && ! grep -qx 'null' "${POD_JSON_FILE}"; then
    POD_NAME=$(jq -r '.metadata.name' < "${POD_JSON_FILE}")
    echo "Found idle pod: ${POD_NAME} — attempting to claim for ${JOB_NAME}..."

    # Build timestamp for annotation
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Patch labels and annotations using jq (POD_JSON from the list query
    # already contains resourceVersion — no second kubectl get needed)
    jq \
      --arg job_name "${JOB_NAME}" \
      --arg timestamp "${TIMESTAMP}" \
      --arg task_type "${TASK_TYPE}" \
      '.metadata.labels.status = "busy"
      | .metadata.annotations["ci.primatrix/claimed-by"] = $job_name
      | .metadata.annotations["ci.primatrix/claimed-at"] = $timestamp
      | .metadata.annotations["ci.primatrix/task-type"] = $task_type' \
      < "${POD_JSON_FILE}" > "${PATCHED_JSON_FILE}"

    # Attempt atomic replace (resourceVersion prevents concurrent claims)
    if kubectl replace -f "${PATCHED_JSON_FILE}" 2>/dev/null; then
      echo "Acquired pod ${POD_NAME}"
      export POD="${POD_NAME}"
      break
    else
      echo "Conflict claiming ${POD_NAME} (another workflow may have taken it); retrying..." >&2
      sleep 2
      continue
    fi
  fi

  # No idle pod found — check timeout
  if (( ELAPSED >= MAX_WAIT )); then
    echo "ERROR: No idle TPU pod became available after ${MAX_WAIT}s." >&2
    exit 1
  fi

  echo "No idle pod available; waiting ${POLL_INTERVAL}s (${ELAPSED}s / ${MAX_WAIT}s elapsed)..."
  sleep "${POLL_INTERVAL}"
  ELAPSED=$(( ELAPSED + POLL_INTERVAL ))
done
