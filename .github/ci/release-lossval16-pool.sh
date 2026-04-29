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

# 1. Owner check: only release if we still hold the Lease.
holder=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)
if [ "$holder" != "$JOB_NAME" ]; then
  echo "WARN: Lease holder '${holder:-(empty)}' != '$JOB_NAME'; skipping cleanup + Lease release"
  exit 0
fi

# 2. Per-pod cleanup: kill the trainer process group (pgid recorded by run.sh)
#    + wipe scratch dirs. Preserve PID 1 (sleep infinity), /opt/base-env, /opt/tools.
#    NOTE: avoid `pkill -f <pattern>` here — the cleanup shell's own command line
#    contains the pattern as an argument, leading to self-kill (exit 137).
for pod in $POOL_PODS; do
  echo "Cleaning $pod..."
  kubectl exec "$pod" -c jax-tpu -- bash -c '
    set +e
    PGID=$(cat "/tmp/ramdisk/runs/'"$RUN_ID"'/pgid" 2>/dev/null)
    if [ -n "$PGID" ] && [ "$PGID" -gt 1 ] 2>/dev/null; then
      # SIGTERM first to let JAX run its TPU cleanup hooks (releases
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
    # Belt-and-suspenders: any orphaned trainer children (reparented to PID 1)
    # whose PPID is now 1 are matched by name, not by command line — process
    # name is "python3", which is too broad, so we instead match by /proc/N/exe
    # symlink containing /opt/ci-env (our per-run venv).
    for pid in $(ls /proc 2>/dev/null | grep -E "^[0-9]+$"); do
      [ -L "/proc/$pid/exe" ] && readlink -f "/proc/$pid/exe" 2>/dev/null \
        | grep -q "^/opt/ci-env/" \
        && kill -9 "$pid" 2>/dev/null
    done
    find /workspace /opt/ci-env -mindepth 1 -delete 2>/dev/null
    rm -rf "/tmp/ramdisk/runs/'"$RUN_ID"'" 2>/dev/null
    rm -rf /tmp/ramdisk/.cache 2>/dev/null
    # libtpu lockfile lingers when JAX is killed via SIGKILL (clean exit
    # would have removed it). Without this, the next runs JAX init throws
    # "ABORTED: Internal error when accessing libtpu multi-process lockfile".
    rm -f /tmp/libtpu_lockfile /tmp/libtpu*.lock /tmp/libtpu_*  2>/dev/null
    true
  ' || echo "  (cleanup on $pod returned non-zero; some state may persist)"
done

# 3. Release Lease (set holderIdentity to null). Track patch failure so we
#    can surface it as workflow-level error rather than warning.
patch_ok=0
if kubectl patch lease "$LEASE" --type=json -p '[
  {"op":"replace","path":"/spec/holderIdentity","value":null}
]' >/dev/null 2>&1; then
  echo "Lease released"
  patch_ok=1
else
  echo "::error::Lease patch failed"
fi

# 4. Readback verification. A successful-looking run with a leaked Lease
#    blocks the next workflow's acquire (until the 5min stale window
#    elapses). Surface this as workflow error so CI is visibly broken.
new_holder=$(kubectl get lease "$LEASE" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null || true)
if [ -n "$new_holder" ] && [ "$new_holder" = "$JOB_NAME" ]; then
  echo "::error::Lease release did not take effect; still held by $JOB_NAME"
  exit 1
fi
if [ "$patch_ok" != "1" ]; then
  exit 1
fi
echo "=== Release complete (lease holder now: '${new_holder:-(empty)}') ==="
