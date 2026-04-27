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
# Use JSON Merge Patch so we only send the fields we want to change. The
# previous `kubectl replace -f` path embedded the pod's resourceVersion
# and PUT the whole object, which routinely 409'd because kubelet updates
# pod status (containerStatuses, probe state) every few seconds — leaving
# stale "busy" claims whose Release step still silently exited 0.
PATCH_BODY='{"metadata":{"labels":{"status":"idle"},"annotations":{"ci.primatrix/claimed-by":null,"ci.primatrix/claimed-at":null,"ci.primatrix/task-type":null}}}'

if ! kubectl patch pod "${POD}" --type=merge -p "${PATCH_BODY}"; then
  echo "::warning::kubectl patch failed on pod ${POD}; claim may be stale" >&2
fi

# --- Readback verification ---
# A zero exit from patch is not sufficient — network or auth flakes can
# still leave the claim in place. Read live state back so infra issues
# surface as GitHub warnings instead of silently leaking a pool slot.
#
# We only flag a leak when the claim is *still our own JOB_NAME* — anything
# else (empty/idle, or claimed-by another job that won the acquire race
# milliseconds after our patch) means our release succeeded. The acquire
# loop polls every 15s and can legitimately reclaim a freshly-released pod
# before this readback runs; treating that as a leak would create noisy
# false-positive infra alerts.
read -r actual_status actual_claim <<< "$(kubectl get pod "${POD}" \
  -o jsonpath='{.metadata.labels.status}{" "}{.metadata.annotations.ci\.primatrix/claimed-by}' \
  2>/dev/null || true)"
if [[ "${actual_claim:-}" == "${JOB_NAME}" ]]; then
  echo "::warning::Pod ${POD} release did not take effect (status='${actual_status}', claimed-by still '${actual_claim}'); pool capacity leaked one slot" >&2
fi

echo "Released pod ${POD}"
