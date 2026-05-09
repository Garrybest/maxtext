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

# --- Owner check (with retries on transient kubectl failures) ---
# A previous version captured `kubectl get` with `2>/dev/null || true` and
# treated empty output the same as "not our pod". GHA-runner→GKE kubectl
# calls flake often enough (auth refresh, API server pressure during cluster
# reconciliation) that this routinely produced empty owner reads, the script
# silently exit 0'd, and pods leaked their `status=busy` claim until the
# 165-min watchdog reaped them — capping effective pool concurrency.
#
# The fix below distinguishes three cases and fails loud (or attempts the
# idempotent merge patch anyway) instead of silently skipping:
#   - kubectl read OK, owner == JOB_NAME    → proceed with release
#   - kubectl read OK, owner == ""          → already released; exit 0
#   - kubectl read OK, owner == other-job   → not ours; exit 0 with WARNING
#   - kubectl read fails 3× in a row        → emit ::warning::, fall through
#                                             to patch (idempotent, mutates
#                                             only annotations we set; the
#                                             readback at the bottom will
#                                             still flag any real leak).
fetch_owner() {
  kubectl get pod "${POD}" \
    -o jsonpath='{.metadata.annotations.ci\.primatrix/claimed-by}' 2>/dev/null
}

actual_owner=""
fetch_ok=0
for attempt in 1 2 3; do
  if owner_out=$(fetch_owner); then
    actual_owner="${owner_out}"
    fetch_ok=1
    break
  fi
  echo "kubectl get pod ${POD} failed (attempt ${attempt}/3); retrying..." >&2
  sleep 2
done

if [[ ${fetch_ok} -eq 1 ]]; then
  if [[ -z "${actual_owner}" ]]; then
    echo "Pod ${POD} has no claim annotation; nothing to release"
    exit 0
  fi
  if [[ "${actual_owner}" != "${JOB_NAME}" ]]; then
    echo "WARNING: pod ${POD} is owned by '${actual_owner}', not '${JOB_NAME}' — skipping release"
    exit 0
  fi
else
  echo "::warning::Could not verify owner for pod ${POD} after 3 retries; attempting patch anyway" >&2
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
