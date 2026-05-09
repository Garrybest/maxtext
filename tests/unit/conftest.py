"""Local conftest for unit tests — stubs heavy deps not available on macOS dev machines."""

import sys
import types

# Stub pathwaysutils (TPU-only)
if "pathwaysutils" not in sys.modules:
  pw = types.ModuleType("pathwaysutils")
  pw.initialize = lambda: None
  sys.modules["pathwaysutils"] = pw

# Stub tensorflow when not installed or broken (not needed for optimizer/fork tests)
try:
  import tensorflow  # noqa: F401  # pylint: disable=unused-import
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
