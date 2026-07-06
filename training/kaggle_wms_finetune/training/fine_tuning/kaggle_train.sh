#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/kaggle_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/kaggle_env.sh"
fi

python "${SCRIPT_DIR}/train_wms.py" \
  --data-path "${SCRIPT_DIR}/data/wms_data_enriched.jsonl" \
  --output-dir /kaggle/working/wms_checkpoints \
  --adapter-dir /kaggle/working/wms_final_adapter \
  --merged-model-dir /kaggle/working/wms_final_model \
  "$@"
