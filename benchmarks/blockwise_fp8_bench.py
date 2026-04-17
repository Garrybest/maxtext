"""Benchmark: Blockwise FP8 vs BF16 DenseGeneral performance.

Measures forward and forward+backward times for jax.lax.dot_general (BF16)
vs BlockwiseFp8DotGeneralOp across realistic DenseGeneral workloads from
ALModel and DeepSeek V3 architectures.

Must run on TPU hardware (FP8 Pallas kernels require native TPU support).
"""

import datetime

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox.blockwise_fp8 import BlockwiseFp8DotGeneralOp


def time_fn(f, *args, warmup=3, iters=10):
  """Time a JAX function, returning (mean_ms, std_ms)."""
  for _ in range(warmup):
    jax.block_until_ready(f(*args))
  times = []
  for _ in range(iters):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    times.append((e - s).total_seconds() * 1000)
  return np.mean(times), np.std(times)


# (label, M, K, N)
WORKLOADS = [
    ("MLP up", 4096, 2048, 5120),
    ("MLP down", 4096, 5120, 2048),
    ("QKV proj", 4096, 2048, 6144),
    ("Attn out", 4096, 2048, 2048),
    ("MLA q_down", 4096, 2048, 256),
    ("MLA kv_down", 4096, 2048, 576),
    ("DS-V3 MLP", 4096, 7168, 18432),
    ("Square 4K", 4096, 4096, 4096),
]

DIMENSION_NUMBERS = (((1,), (0,)), ((), ()))


def run_benchmarks():
  """Run FP8 vs BF16 benchmarks across all workloads."""
  print(f"Device: {jax.devices()[0]}")
  print(f"Num devices: {jax.device_count()}")
  print()

  # Header
  header = (
      f"{'Workload':<14} {'M':>5} {'K':>5} {'N':>6}"
      f" | {'BF16 fwd':>10} {'FP8 fwd':>10} {'Speedup':>7}"
      f" | {'BF16 TFLOP/s':>12} {'FP8 TFLOP/s':>12}"
      f" | {'BF16 fwd+bwd':>12} {'FP8 fwd+bwd':>12} {'Speedup':>7}"
  )
  print(header)
  print("-" * len(header))

  for label, M, K, N in WORKLOADS:
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    lhs = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (K, N), dtype=jnp.bfloat16)

    flops_fwd = 2.0 * M * K * N

    # -- BF16 forward --
    @jax.jit
    def bf16_fwd(l, r):
      return jax.lax.dot_general(l, r, DIMENSION_NUMBERS, preferred_element_type=jnp.float32)

    bf16_fwd_ms, _ = time_fn(bf16_fwd, lhs, rhs)

    # -- FP8 forward --
    op = BlockwiseFp8DotGeneralOp(block_size=128)

    @jax.jit
    def fp8_fwd(l, r):
      return op(l, r, DIMENSION_NUMBERS)  # pylint: disable=cell-var-from-loop

    fp8_fwd_ms, _ = time_fn(fp8_fwd, lhs, rhs)

    # -- BF16 forward+backward --
    @jax.jit
    def bf16_fwd_bwd(l, r):
      def loss(l, r):
        return jnp.sum(jax.lax.dot_general(l, r, DIMENSION_NUMBERS, preferred_element_type=jnp.float32))

      return jax.grad(loss, argnums=(0, 1))(l, r)

    bf16_fb_ms, _ = time_fn(bf16_fwd_bwd, lhs, rhs)

    # -- FP8 forward+backward --
    fp8_fb_ms = None
    try:

      @jax.jit
      def fp8_fwd_bwd(l, r):
        def loss(l, r):
          return jnp.sum(op(l, r, DIMENSION_NUMBERS))  # pylint: disable=cell-var-from-loop

        return jax.grad(loss, argnums=(0, 1))(l, r)

      fp8_fb_ms, _ = time_fn(fp8_fwd_bwd, lhs, rhs)
    except Exception as e:  # pylint: disable=broad-except
      print(f"  [WARN] FP8 fwd+bwd failed for {label}: {e}")

    # Compute TFLOP/s
    bf16_tflops = flops_fwd / (bf16_fwd_ms / 1000) / 1e12
    fp8_tflops = flops_fwd / (fp8_fwd_ms / 1000) / 1e12

    fwd_speedup = bf16_fwd_ms / fp8_fwd_ms

    if fp8_fb_ms is not None:
      fb_speedup = bf16_fb_ms / fp8_fb_ms
      fb_str = f" | {bf16_fb_ms:>9.2f} ms {fp8_fb_ms:>9.2f} ms {fb_speedup:>6.2f}x"
    else:
      fb_str = f" | {bf16_fb_ms:>9.2f} ms {'ERR':>9} ms {'N/A':>7}"

    print(
        f"{label:<14} {M:>5} {K:>5} {N:>6}"
        f" | {bf16_fwd_ms:>7.2f} ms {fp8_fwd_ms:>7.2f} ms {fwd_speedup:>6.2f}x"
        f" | {bf16_tflops:>9.1f} TF/s {fp8_tflops:>9.1f} TF/s"
        f"{fb_str}"
    )

  print()
  print("Speedup > 1.0 means FP8 is faster than BF16.")


if __name__ == "__main__":
  run_benchmarks()
