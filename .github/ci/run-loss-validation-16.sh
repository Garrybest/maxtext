#!/usr/bin/env bash
# run-loss-validation-16.sh — per-pod entrypoint for the loss-validation run.
#
# Invoked once per pod by the workflow:
#   kubectl exec POD -c jax-tpu -- bash -c "
#     setsid bash /workspace/maxtext/.github/ci/run-loss-validation-16.sh $RUN_ID \
#       < /dev/null > /dev/null 2>&1 & disown
#   "
# `setsid` + `disown` + `< /dev/null` makes the trainer survive the kubectl
# exec stream closing. The script writes its log to $RUN_DIR/log and its exit
# code to $RUN_DIR/exit_code, then drops $RUN_DIR/done. Workflow polls these.
#
# All training env (STEPS, parallelism, GCS_BUCKET, etc.) comes from
# $RUN_DIR/run.env, written by the workflow before launch. The
# `RUN_ID_INSIDE_ENV` field is checked here to reject stale env from a
# previous run.
set -uo pipefail

RUN_ID="${1:?RUN_ID arg required}"
RUN_DIR="/tmp/ramdisk/runs/$RUN_ID"

# Always-runs cleanup: any exit path writes exit_code + done. The poll loop
# in the workflow uses these files to determine completion; a missing
# exit_code causes the poll to wait forever (until the 130min hard timeout).
finalize() {
  rc=$?
  echo "$rc" > "$RUN_DIR/exit_code" 2>/dev/null || true
  touch "$RUN_DIR/done" 2>/dev/null || true
  exit $rc
}
trap finalize EXIT

if [ ! -f "$RUN_DIR/run.env" ]; then
  # Write to log (not stdout — launch redirects stdout to /dev/null).
  echo "ERROR: missing $RUN_DIR/run.env" >> "$RUN_DIR/log" 2>/dev/null || true
  exit 1
fi

# Load env (source after `set -a` so variables become exported).
set -a
. "$RUN_DIR/run.env"
set +a

# Reject stale env from a previous run.
if [ "${RUN_ID_INSIDE_ENV:-}" != "$RUN_ID" ]; then
  echo "ERROR: RUN_ID mismatch (env says '${RUN_ID_INSIDE_ENV:-}', arg '$RUN_ID')" >> "$RUN_DIR/log" 2>/dev/null || true
  exit 1
fi

# Record our PID as the process-group ID for clean teardown by release.
# Workflow launches us via `setsid bash run.sh ...`, which makes us a session
# leader; our PID == our PGID. Release uses `kill -- -$PGID` to terminate the
# whole subtree (trainer + venv installs + python) atomically — avoids the
# `pkill -f` self-kill pitfall when the cleanup shell's own command line
# happens to contain a substring like "pretrain_ling2".
echo "$$" > "$RUN_DIR/pgid"

# Pod index: prefer Downward-API label (apps.kubernetes.io/pod-index, K8s ≥1.28),
# fall back to suffix of POD_NAME. Validate numeric.
POD_INDEX="${POD_INDEX:-${POD_NAME##*-}}"
case "$POD_INDEX" in
  ''|*[!0-9]*)
    echo "ERROR: POD_INDEX not numeric ('$POD_INDEX')" >> "$RUN_DIR/log" 2>/dev/null || true
    exit 1
    ;;
esac

cd /workspace/maxtext

# Subshell with `set -e` so any install/training failure exits with that
# rc. Without -e, an early failure (e.g. uv pip install) would be masked
# by the trailing `if [ ... ]` clause that returns 0 on pods 1..3 — false
# green per pod. The trap captures rc on subshell exit and writes
# exit_code/done.
( set -euo pipefail
  echo "=== run-loss-validation-16.sh ==="
  echo "POD_NAME=$POD_NAME POD_INDEX=$POD_INDEX RUN_ID=$RUN_ID"
  date -u +%Y-%m-%dT%H:%M:%SZ

  # Per-run venv layered on /opt/base-env (jax[tpu] is already pre-installed
  # in /opt/base-env by the StatefulSet's initContainer; we extend it with
  # maxtext + repo-local TPU deps).
  /opt/tools/uv venv --python 3.12 --seed --system-site-packages /opt/ci-env
  . /opt/ci-env/bin/activate

  # Cold-start auth for `uv pip install`'s transitive private git deps
  # (`tops @ git+primatrix/pallas-kernel`, `tokamax @ git+primatrix/tokamax`).
  # Warm runs short-circuit via /tmp/ramdisk/.cache/uv git cache and never
  # hit the network; cold-start paths (pod restart, livenessProbe failover,
  # manual pool wipe) clear tmpfs and reach this fetch — failing 401 without
  # auth. GITHUB_TOKEN comes from $RUN_DIR/run.env (CROSS_REPO_TOKEN with
  # github.token fallback). GIT_CONFIG_GLOBAL=tmpfile + trap mirrors the
  # workflow's Stage step so the token-bearing `insteadOf` rule does not
  # persist in /root/.gitconfig.
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    export GIT_CONFIG_GLOBAL=$(mktemp)
    trap 'rm -f "${GIT_CONFIG_GLOBAL:-}"' EXIT
    git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
  fi

  /opt/tools/uv pip install -e '.[tpu]' --resolution=lowest

  # optax from source (matches the existing CI Job yaml).
  pip uninstall optax -y 2>/dev/null || true
  pip install git+https://github.com/google-deepmind/optax

  # Run training. Loss values written under $RUN_DIR (per-run scope) so a
  # crashed run doesn't pollute the next run's gate input.
  bash scripts/pretrain_ling2.sh \
    metrics_file="$RUN_DIR/loss_validation_metrics.txt"

  # Pod-0 runs the threshold gate against the in-repo reference JSON.
  if [ "$POD_INDEX" = "0" ]; then
    python3 scripts/compare_loss_100step.py "$RUN_DIR/loss_validation_metrics.txt"
  fi
) >> "$RUN_DIR/log" 2>&1
# trap finalize EXIT writes exit_code/done with this script's exit rc.
exit $?
