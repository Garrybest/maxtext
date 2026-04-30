#!/usr/bin/env bash
# release-lossval16-pool.sh — release the 16-chip pool back to idle.
#
# Always-runs cleanup. Owner check on Lease prevents accidentally releasing
# a pool another workflow holds (e.g. concurrent dispatch confusion).
#
# Required env:
#   JOB_NAME — must match Lease holderIdentity to do real work
#   RUN_ID   — per-run dir to wipe
set -uo pipefail

: "${JOB_NAME:?JOB_NAME env var is required}"
: "${RUN_ID:?RUN_ID env var is required}"
LEASE=ci-lossval16-pool-claim
POOL_PODS="ci-lossval16-pool-0 ci-lossval16-pool-1 ci-lossval16-pool-2 ci-lossval16-pool-3"

echo "=== Lossval16 pool release (job=$JOB_NAME run=$RUN_ID) ==="

# 1. Owner check with retry tolerance. Up to 3 retries (~10s) to confirm
#    the holder. CLEANUP_MODE governs which paths we touch:
#      full    — kill our pgid + wipe shared paths (/workspace, /opt/ci-env,
#                /tmp/ramdisk/.cache, /tmp/libtpu*). Only safe when we
#                confidently still hold the Lease.
#      minimal — kill our pgid + remove our RUN_ID dir only. Safe even when
#                ownership transitioned (pgid is process-tree-local, RUN_ID
#                dir is unique per workflow run). Never touches shared paths
#                that another workflow may be using.
#    A failed Lease readback (kubectl unreachable) downgrades to minimal —
#    we won't wipe /workspace under uncertainty.
attempts=0
holder=""
kubectl_ok=0
while [ $attempts -lt 3 ]; do
  if holder=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null); then
    kubectl_ok=1
    break
  fi
  attempts=$((attempts + 1))
  sleep 5
done
if [ "$kubectl_ok" = "1" ] && [ "$holder" = "$JOB_NAME" ]; then
  CLEANUP_MODE=full
  echo "Owner confirmed (holder=$holder); CLEANUP_MODE=full"
elif [ "$kubectl_ok" = "1" ]; then
  # Holder is empty (Lease already released) or another workflow's. Either
  # way we don't own the Lease — refuse shared-path cleanup that could
  # affect a concurrent run. Skip Lease release attempt entirely.
  echo "WARN: Lease holder '${holder:-(empty)}' != '$JOB_NAME'; skipping cleanup + Lease release"
  exit 0
else
  # kubectl unreachable after 3 retries. Don't gamble on shared paths.
  CLEANUP_MODE=minimal
  echo "WARN: kubectl get lease failed 3x; CLEANUP_MODE=minimal (RUN_ID-scoped only)"
fi

# 2. Per-pod cleanup: kill the trainer process group (pgid recorded by run.sh)
#    + wipe scratch dirs. Preserve PID 1 (sleep infinity), /opt/base-env, /opt/tools.
#    NOTE: avoid `pkill -f <pattern>` here — the cleanup shell's own command line
#    contains the pattern as an argument, leading to self-kill (exit 137).
for pod in $POOL_PODS; do
  echo "Cleaning $pod ($CLEANUP_MODE)..."
  kubectl exec "$pod" -c jax-tpu -- bash -c '
    set +e
    PGID=$(cat "/tmp/ramdisk/runs/'"$RUN_ID"'/pgid" 2>/dev/null)
    if [ -n "$PGID" ] && [ "$PGID" -gt 1 ] 2>/dev/null; then
      # SIGTERM first so JAX runs its TPU cleanup hooks (releases
      # /dev/vfio/<group> IOMMU binding cleanly). SIGKILL after a short
      # grace period only if the trainer ignored SIGTERM. Pure SIGKILL
      # leaves the vfio group bound, blocking the next JAX init with
      # "Device or resource busy" until the pod is recreated.
      kill -TERM -- "-$PGID" 2>/dev/null
      for _ in 1 2 3 4 5 6 7 8; do
        kill -0 -- "-$PGID" 2>/dev/null || break
        sleep 1
      done
      kill -9 -- "-$PGID" 2>/dev/null
    fi
    # RUN_ID-scoped cleanup: always safe (unique per workflow run).
    rm -rf "/tmp/ramdisk/runs/'"$RUN_ID"'" 2>/dev/null

    if [ "'"$CLEANUP_MODE"'" = "full" ]; then
      # Belt-and-suspenders: any orphaned trainer children (reparented to PID 1)
      # whose PPID is now 1 are matched by /proc/N/exe symlink containing
      # /opt/ci-env (our per-run venv). Skipped under minimal mode because
      # /opt/ci-env may belong to a concurrent workflow that took the Lease.
      for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
        [ -L "/proc/$pid/exe" ] && readlink -f "/proc/$pid/exe" 2>/dev/null \
          | grep -q "^/opt/ci-env/" \
          && kill -9 "$pid" 2>/dev/null
      done
      # Shared-path wipes — only safe when ownership confirmed.
      find /workspace /opt/ci-env -mindepth 1 -delete 2>/dev/null
      rm -rf /tmp/ramdisk/.cache 2>/dev/null
      # libtpu lockfile lingers when JAX is SIGKILL'"'"'d (clean exit would
      # have removed it). Without this the next JAX init throws "ABORTED:
      # Internal error when accessing libtpu multi-process lockfile".
      rm -f /tmp/libtpu_lockfile /tmp/libtpu*.lock /tmp/libtpu_*  2>/dev/null
    fi
    true
  ' || echo "  (cleanup on $pod returned non-zero; some state may persist)"
done

# 3. Atomic conditional Lease release. test+replace in one round-trip:
#    - if we still hold (normal): test passes, replace nulls holderIdentity → success
#    - if we lost ownership (rare stale-takeover): test fails, replace skipped → no-op
#    - if kubectl errors transiently: patch fails → fall through to readback below
#    This eliminates the prior race where an unconditional `replace` could yank
#    a takeover's lease, AND makes "we don't hold" a no-op rather than a steal.
if kubectl patch lease "$LEASE" --type=json -p '[
  {"op":"test","path":"/spec/holderIdentity","value":"'"$JOB_NAME"'"},
  {"op":"replace","path":"/spec/holderIdentity","value":null}
]' >/dev/null 2>&1; then
  echo "Lease released"
  exit 0
fi

# 4. Readback to determine why patch failed. Retry 3× to absorb transient API
#    hiccups. Track readback_ok separately from new_holder: empty new_holder
#    after a SUCCESSFUL kubectl call means "Lease has no holder" (success),
#    but empty after FAILED kubectl calls means "we don't know" (must surface
#    as error — a leaked Lease blocks the next acquire until stale window).
readback_ok=0
new_holder=""
for _ in 1 2 3; do
  if new_holder=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null); then
    readback_ok=1
    break
  fi
  sleep 5
done
if [ "$readback_ok" = "0" ]; then
  echo "::error::Lease patch failed and readback verification failed (kubectl unreachable); Lease state unknown — manual intervention may be needed"
  exit 1
fi
if [ "$new_holder" = "$JOB_NAME" ]; then
  echo "::error::Lease patch failed and we still hold (holder='$new_holder'); next acquire will spin until stale window"
  exit 1
fi
echo "=== Release complete (lease holder now: '${new_holder:-(empty)}'; patch test op rejected — ownership had moved) ==="
