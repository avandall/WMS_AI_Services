from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_WMS_PATH = ROOT_DIR / "training" / "fine_tuning" / "train_wms.py"
spec = importlib.util.spec_from_file_location("train_wms_for_test", TRAIN_WMS_PATH)
assert spec is not None and spec.loader is not None
train_wms = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_wms
spec.loader.exec_module(train_wms)


def test_training_template_converts_sql_output_to_runtime_json_shape() -> None:
    question = train_wms.build_question(
        instruction="Tìm tồn kho của SKU-001.",
        input_text="Bảng: inventory.",
    )

    template = train_wms.build_training_template(
        question=question,
        output="```sql\nSELECT quantity FROM inventory WHERE sku = 'SKU-001' LIMIT 1;\n```",
    )

    assert template == {
        "intent": "inventory_lookup",
        "target": "inventory",
        "filters": {"sku": "SKU-001"},
        "metrics": ["quantity"],
        "limit": 1,
        "sql": "SELECT quantity FROM inventory WHERE sku = 'SKU-001' LIMIT 1;",
    }


def test_training_prompt_matches_runtime_query_planner_prompt() -> None:
    text = train_wms.formatting_prompts_func(
        {
            "instruction": ["Tìm tồn kho của SKU-001."],
            "input": ["Bảng: inventory."],
            "output": ["SELECT quantity FROM inventory WHERE sku = 'SKU-001';"],
        }
    )["text"][0]

    assert text.startswith("You are a WMS query planner.")
    assert '"intent": "inventory_lookup"' in text
    assert '"sql": "SELECT quantity FROM inventory WHERE sku = \'SKU-001\';"' in text


def test_enriched_dataset_is_valid_and_covers_core_wms_domains() -> None:
    dataset_path = ROOT_DIR / "training" / "fine_tuning" / "data" / "wms_data_enriched.jsonl"
    intents: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    rows = 0

    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        assert set(row) == {"instruction", "input", "output"}
        payload = json.loads(row["output"])
        assert isinstance(payload["filters"], dict), line_number
        assert isinstance(payload["metrics"], list), line_number
        assert {"intent", "target", "filters", "metrics", "limit", "sql"} <= set(payload)
        intents[payload["intent"]] += 1
        targets[payload["target"]] += 1
        rows += 1

    assert rows >= 600
    assert {
        "inventory_lookup",
        "document_lookup",
        "warehouse_lookup",
        "product_lookup",
        "customer_lookup",
        "report_lookup",
        "order_status",
        "unknown",
    } <= set(intents)
    assert {
        "inventory",
        "documents",
        "positions",
        "warehouses",
        "products",
        "customers",
        "reporting",
        "orders",
        "unknown",
    } <= set(targets)
