#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python - <<'PY'
import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA runtime:", getattr(torch.version, "cuda", None))
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", type(exc).__name__, exc)
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y torchcodec sentence-transformers || true
python -m pip install --no-cache-dir \
  -r "${SCRIPT_DIR}/requirements.kaggle-unsloth.txt" \
  -c "${SCRIPT_DIR}/constraints.kaggle-unsloth.txt"

NVIDIA_LIB_PATH="$(
python - <<'PY'
from pathlib import Path
import site

site_dirs = [Path(path) for path in site.getsitepackages()]
user_site = site.getusersitepackages()
if user_site:
    site_dirs.append(Path(user_site))

paths = []
for site_dir in site_dirs:
    nvidia_dir = site_dir / "nvidia"
    if not nvidia_dir.exists():
        continue
    paths.extend(str(path) for path in sorted(nvidia_dir.glob("*/lib")) if path.is_dir())
print(":".join(paths))
PY
)"

if [[ -n "${NVIDIA_LIB_PATH}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  cat > "${SCRIPT_DIR}/kaggle_env.sh" <<EOF
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
fi

python - <<'PY'
import inspect

import unsloth
import transformers
import trl
from trl import SFTConfig, SFTTrainer

print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
print("unsloth:", getattr(unsloth, "__version__", "unknown"))
print("SFTConfig.max_length:", "max_length" in inspect.signature(SFTConfig).parameters)
print("SFTTrainer.processing_class:", "processing_class" in inspect.signature(SFTTrainer).parameters)
PY
