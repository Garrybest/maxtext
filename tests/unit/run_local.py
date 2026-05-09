#!/usr/bin/env python3
"""Lightweight local test runner for optimizer/fork tests on macOS.

Usage: PYTHONCASEOK=1 python tests/unit/run_local.py [test_pattern]

Stubs heavy dependencies (tensorflow, pathwaysutils) that are unavailable
on macOS dev machines, then runs the specified test(s).

Do NOT rename this file to match pytest's `python_files` glob (e.g.
`*_test.py` or `*_tests.py`). If pytest collects this as a test module,
its module-body side effects (`os.environ.setdefault("DECOUPLE_GCLOUD",
"TRUE")`, `sys.modules` stubs) leak into the whole pytest session and
silently break real tests — notably `custom_mesh_and_rule_test.py`, which
then picks up `decoupled_base_test.yml` with a hardcoded
`ici_fsdp_parallelism: 1`.
"""
import sys
import types
import os

# --- stubs before any project import ---
os.environ.setdefault("DECOUPLE_GCLOUD", "TRUE")

pw = types.ModuleType("pathwaysutils")
pw.initialize = lambda: None
sys.modules["pathwaysutils"] = pw

try:
  import tensorflow  # pylint: disable=unused-import
except (ImportError, AttributeError, TypeError):
  tf = types.ModuleType("tensorflow")
  tf_data = types.ModuleType("tensorflow.data")
  tf_data.Dataset = type("Dataset", (), {})
  tf.data = tf_data
  tf_io = types.ModuleType("tensorflow.io")
  tf_gfile = types.ModuleType("tensorflow.io.gfile")
  tf_gfile.exists = lambda x: False
  tf_gfile.GFile = type("GFile", (), {"__init__": lambda s, *a, **k: None})
  tf_io.gfile = tf_gfile
  tf.io = tf_io
  tf.TensorSpec = type("TensorSpec", (), {})
  for name, mod in [
      ("tensorflow", tf),
      ("tensorflow.data", tf_data),
      ("tensorflow.io", tf_io),
      ("tensorflow.io.gfile", tf_gfile),
  ]:
    sys.modules[name] = mod

# --- end stubs ---

if __name__ == "__main__":
  import pytest

  args = ["-xvs", "--no-header", "--tb=short", "-p", "no:cacheprovider"]
  args.extend(sys.argv[1:] or ["tests/unit/optimizers_test.py", "tests/unit/optax_muon_fork_test.py"])
  raise SystemExit(pytest.main(args))
