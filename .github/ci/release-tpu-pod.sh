#!/usr/bin/env bash
set -uo pipefail

# Release a CI TPU pod back to the idle pool, with owner verification.
# Called from GitHub Actions `if: always()` cleanup steps.
#
# Required env vars:
#   POD       - name of the pod to release
#   JOB_NAME  - GitHub Actions job name that claimed the pod
#
# Usage (sourced or executed):
#   source .github/ci/release-tpu-pod.sh
#   # or
#   bash .github/ci/release-tpu-pod.sh

# --- Validate required env vars ---
if [[ -z "${POD:-}" ]]; then
  echo "ERROR: POD env var is required" >&2
  exit 1
fi

if [[ -z "${JOB_NAME:-}" ]]; then
  echo "ERROR: JOB_NAME env var is required" >&2
  exit 1
fi

echo "Releasing pod ${POD} (owner: ${JOB_NAME})..."

# --- Owner check ---
actual_owner=$(kubectl get pod "${POD}" \
  -o jsonpath='{.metadata.annotations.ci\.primatrix/claimed-by}' 2>/dev/null || true)

if [[ "${actual_owner}" != "${JOB_NAME}" ]]; then
  echo "WARNING: pod ${POD} is owned by '${actual_owner}', not '${JOB_NAME}' — skipping release"
  exit 0
fi

# --- Kill user processes (preserve sleep infinity and PID 1) ---
kubectl exec "${POD}" -- bash -c \
  'pgrep -v -f "sleep infinity" | grep -v "^1$" | xargs -r kill -9 2>/dev/null || true' \
  || true

# --- Clean workspace ---
kubectl exec "${POD}" -- find /workspace /opt/ci-env -mindepth 1 -delete \
  2>/dev/null || true

# --- Atomic release: set status=idle and remove claim annotations ---
# Use temp files to avoid bash variable mangling JSON escape sequences.
RELEASE_JSON_FILE=$(mktemp)
RELEASE_PATCHED_FILE=$(mktemp)
trap 'rm -f "${RELEASE_JSON_FILE}" "${RELEASE_PATCHED_FILE}"' EXIT

kubectl get pod "${POD}" -o json > "${RELEASE_JSON_FILE}" 2>/dev/null || true
if [[ -s "${RELEASE_JSON_FILE}" ]]; then
  jq '
      .metadata.labels.status = "idle"
      | del(.metadata.annotations["ci.primatrix/claimed-by"])
      | del(.metadata.annotations["ci.primatrix/claimed-at"])
      | del(.metadata.annotations["ci.primatrix/task-type"])
    ' < "${RELEASE_JSON_FILE}" > "${RELEASE_PATCHED_FILE}" \
    && kubectl replace -f "${RELEASE_PATCHED_FILE}" \
    || echo "WARNING: kubectl replace failed; pod may not have been reset to idle" >&2
else
  echo "WARNING: pod ${POD} not found; skipping label reset" >&2
fi

echo "Released pod ${POD}"
