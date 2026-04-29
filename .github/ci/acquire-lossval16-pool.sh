#!/usr/bin/env bash
# acquire-lossval16-pool.sh — claim the 16-chip CI shell pool via a Lease.
#
# Hard pre-checks:
#   - all 4 pool pods Ready
#   - StatefulSet `ci-lossval16-pool` exists
# Then atomically take the Lease via JSON-patch test+set on holderIdentity.
# Sets up a per-run scratch dir on each pod.
#
# Required env vars:
#   JOB_NAME — unique workflow identity (becomes Lease holder)
# Optional:
#   GITHUB_RUN_NUMBER, GITHUB_RUN_ATTEMPT — used to derive RUN_ID
#   POOL_ACQUIRE_TIMEOUT — seconds to wait for Lease (default 7200)
#
# Exports on success:
#   RUN_ID    — unique scratch-dir name (e.g., run-123-1)
#   POOL_PODS — "ci-lossval16-pool-0 ci-lossval16-pool-1 ci-lossval16-pool-2 ci-lossval16-pool-3"
set -euo pipefail

: "${JOB_NAME:?JOB_NAME env var is required}"
LEASE=ci-lossval16-pool-claim
STS=ci-lossval16-pool
# Compute RUN_ID up-front so the Lease annotation in the claim loop can
# include it for oncall triage. Same value used later in step 3.
RUN_ID="run-${GITHUB_RUN_NUMBER:-$$}-${GITHUB_RUN_ATTEMPT:-1}"
TIMEOUT=${POOL_ACQUIRE_TIMEOUT:-7200}
READY_TIMEOUT=${POOL_READY_TIMEOUT:-1800}   # 30min default (fail-fast). Covers warm-pool happy path
                                            # (pods already Ready, returns instantly) and a single
                                            # NAP nodepool create + initContainer cycle (~15-20min).
                                            # When zone capacity is tight beyond 30min, fail and let
                                            # the next cron tick retry (StatefulSet keeps the
                                            # self-heal loop going in the background regardless).
                                            # Override via POOL_READY_TIMEOUT env for stress tests.
POOL_PODS="ci-lossval16-pool-0 ci-lossval16-pool-1 ci-lossval16-pool-2 ci-lossval16-pool-3"
_self="${BASH_SOURCE[0]:-${0:-}}"
if [ -n "$_self" ] && [ -f "$_self" ]; then
  POOL_YAML="$(dirname "$_self")/tpu-lossval16-pool.yaml"
else
  # Fall back: assume CWD is the repo root (CI workflow invocation pattern).
  POOL_YAML=".github/ci/tpu-lossval16-pool.yaml"
fi

echo "=== Lossval16 pool acquire ==="

# 1. Self-heal: ensure pool resources exist. `kubectl apply` is idempotent —
#    if the StatefulSet/Service/Lease/PDB were deleted (intentionally or
#    accidentally), this recreates them. If they're in place but pods are
#    missing (e.g. `kubectl delete pod --force` or GCE node loss), the
#    StatefulSet controller has already started recreating; we just wait.
#    Underlying nodepool is left to GKE NAP / autoscaler.
if [ -f "$POOL_YAML" ]; then
  echo "Ensuring pool resources from $POOL_YAML..."
  kubectl apply -f "$POOL_YAML" >/dev/null
else
  echo "WARN: $POOL_YAML not found; assuming pool already provisioned"
  kubectl get statefulset "$STS" >/dev/null || {
    echo "ERROR: StatefulSet $STS missing and no yaml to apply" >&2
    exit 1
  }
fi

# 2. Wait for all 4 pods Ready (covers fresh apply, pod recreation, NAP node provision).
echo "Waiting up to ${READY_TIMEOUT}s for $STS to be 4/4 Ready..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while true; do
  ready_count=0
  for pod in $POOL_PODS; do
    ready=$(kubectl get pod "$pod" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)
    [ "$ready" = "True" ] && ready_count=$((ready_count + 1))
  done
  if [ "$ready_count" -eq 4 ]; then
    echo "Pre-check OK: all 4 pods Ready"
    break
  fi
  if [ $(date +%s) -ge $deadline ]; then
    echo "ERROR: ready timeout after ${READY_TIMEOUT}s ($ready_count/4 ready). Pod status:" >&2
    kubectl get pods -l pool=ci-lossval16 -o wide 2>&1 | head -10 >&2
    exit 1
  fi
  echo "  $ready_count/4 ready, retry in 15s ($((deadline - $(date +%s)))s left)"
  sleep 15
done

# 2. Atomic claim via JSON-patch test+set on holderIdentity.
#    The 'test' op confirms current holder is null in the same round-trip
#    as the 'replace' that sets it to JOB_NAME — no TOCTOU.
#
#    Lease expiry: if the current holder's `renewTime` is older than
#    5 × leaseDurationSeconds (default 60s → 5 min stale), assume the
#    previous workflow died without releasing and force-release once.
#    This avoids permanent deadlock when a workflow is killed (e.g.,
#    GitHub runner network loss, manual cancel, OOM) before its
#    if: always() release step runs.
deadline=$(( $(date +%s) + TIMEOUT ))
STALE_AFTER=300   # 5 min: 5x default leaseDurationSeconds
while true; do
  now_iso=$(date -u +%Y-%m-%dT%H:%M:%S.000000Z)
  if kubectl patch lease "$LEASE" --type=json -p "[
    {\"op\":\"test\",\"path\":\"/spec/holderIdentity\",\"value\":null},
    {\"op\":\"replace\",\"path\":\"/spec/holderIdentity\",\"value\":\"$JOB_NAME\"},
    {\"op\":\"replace\",\"path\":\"/spec/acquireTime\",\"value\":\"$now_iso\"},
    {\"op\":\"replace\",\"path\":\"/spec/renewTime\",\"value\":\"$now_iso\"}
  ]" >/dev/null 2>&1; then
    echo "Lease acquired by $JOB_NAME"
    # Best-effort metadata for oncall triage (skipped silently on failure;
    # not part of the atomic claim).
    annot_run_url=""
    if [ -n "${GITHUB_SERVER_URL:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ]; then
      annot_run_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
    fi
    kubectl annotate lease "$LEASE" --overwrite \
      "ci.primatrix/run-url=${annot_run_url}" \
      "ci.primatrix/branch=${GITHUB_REF_NAME:-}" \
      "ci.primatrix/sha=${GITHUB_SHA:-}" \
      "ci.primatrix/run-id-internal=${RUN_ID}" >/dev/null 2>&1 || true
    break
  fi

  # Could not claim — inspect current holder + renew age.
  current=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)
  renew=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.renewTime}' 2>/dev/null || true)
  rv=$(kubectl get lease "$LEASE" -o jsonpath='{.metadata.resourceVersion}' 2>/dev/null || true)
  if [ -n "$renew" ]; then
    renew_epoch=$(date -u -d "$renew" +%s 2>/dev/null || echo 0)
    age=$(( $(date +%s) - renew_epoch ))
    if [ -n "$current" ] && [ "$age" -gt "$STALE_AFTER" ] && [ -n "$rv" ]; then
      # Conditional force-release: JSON-patch tests the resourceVersion we
      # observed; if a concurrent live holder renewed between our get and
      # patch (any field change bumps resourceVersion), the patch fails
      # and we just loop. Avoids racy-takeover of a healthy holder.
      echo "Lease holder '$current' looks stale (${age}s since renew, rv=$rv); attempting conditional force-release."
      if kubectl patch lease "$LEASE" --type=json -p "[
        {\"op\":\"test\",\"path\":\"/metadata/resourceVersion\",\"value\":\"$rv\"},
        {\"op\":\"replace\",\"path\":\"/spec/holderIdentity\",\"value\":null}
      ]" >/dev/null 2>&1; then
        echo "  conditional patch succeeded; retrying claim."
      else
        echo "  conditional patch rejected (holder renewed concurrently); will re-evaluate."
      fi
      continue
    fi
    echo "Lease held by '${current:-?}' (last renew ${age}s ago, deadline in $((deadline - $(date +%s)))s)..."
  else
    echo "Lease held by '${current:-?}' (no renewTime yet, deadline in $((deadline - $(date +%s)))s)..."
  fi

  if [ $(date +%s) -ge $deadline ]; then
    echo "ERROR: ACQUIRE TIMEOUT after ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 15
done

# RUN_ID was computed at script top so the Lease annotation could reference
# it; re-echo here for the workflow log.
echo "RUN_ID=$RUN_ID"

# 3a. Emit RUN_ID to $GITHUB_OUTPUT IMMEDIATELY after Lease is claimed,
#     before any cleanup that could fail. The release step uses RUN_ID to
#     locate per-run scratch; without it, release returns early and the
#     Lease leaks even though we hold it. (Lease ownership is the strict
#     prereq for cleanup; cleanup quality is best-effort.)
export RUN_ID
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "RUN_ID=$RUN_ID" >> "$GITHUB_OUTPUT"
fi
export POOL_PODS

# 3b. Defensive cleanup on each pod.
#    Covers the case where the previous workflow died without running its
#    `if: always()` release step (we just force-released the Lease above).
#    Wipe per-run scratch + workspace + per-run venv + libtpu lockfile, and
#    kill any leftover trainer processes. Preserve /opt/base-env (jax[tpu])
#    and /opt/tools (uv) — those are persistent across runs by design.
#
#    Skip self ($$) and PID 1 from any kill loop to avoid the bash-cmdline
#    self-kill pitfall (an earlier `pkill -f pretrain_ling2` foot-gun).
echo "Defensive cleanup on all 4 pods..."
for pod in $POOL_PODS; do
  kubectl exec "$pod" -c jax-tpu -- bash -c '
    set +e
    self=$$
    # Kill processes whose exe is /opt/ci-env/* (per-run venv leftover).
    for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
      [ "$pid" = "$self" ] && continue
      [ "$pid" = "1" ] && continue
      [ -L "/proc/$pid/exe" ] && readlink -f "/proc/$pid/exe" 2>/dev/null \
        | grep -q "^/opt/ci-env/" \
        && kill -9 "$pid" 2>/dev/null
    done
    # Kill any process group recorded by a previous run that ours did not own.
    for pgidfile in /tmp/ramdisk/runs/*/pgid; do
      [ -f "$pgidfile" ] || continue
      pg=$(cat "$pgidfile" 2>/dev/null)
      [ -n "$pg" ] && [ "$pg" -gt 1 ] 2>/dev/null && kill -9 -- "-$pg" 2>/dev/null
    done
    # Wipe transient state.
    find /workspace /opt/ci-env -mindepth 1 -delete 2>/dev/null
    rm -rf /tmp/ramdisk/runs/* 2>/dev/null
    rm -f /tmp/libtpu_lockfile /tmp/libtpu*.lock /tmp/libtpu_*  2>/dev/null
    true
  ' >/dev/null 2>&1 || true
done
# 3c. Create the fresh per-run dir.
for pod in $POOL_PODS; do
  kubectl exec "$pod" -c jax-tpu -- mkdir -p "/tmp/ramdisk/runs/$RUN_ID" >/dev/null
done
echo "Per-run dirs ready on all 4 pods"
echo "=== Acquire complete ==="
