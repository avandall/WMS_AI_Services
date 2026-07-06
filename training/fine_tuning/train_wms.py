from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import site
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
TEMPLATES_PATH = ROOT_DIR / "src" / "ai_service" / "pipeline" / "templates.py"


def _load_template_helpers():
    spec = importlib.util.spec_from_file_location("wms_ai_templates", TEMPLATES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load template helpers from {TEMPLATES_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_query_template_prompt


build_query_template_prompt = _load_template_helpers()


def main() -> None:
    ensure_nvidia_library_path()
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_path = args.data_path or script_dir / "data" / "wms_data_enriched.jsonl"
    output_dir = args.output_dir or script_dir / "wms_checkpoints"
    adapter_dir = args.adapter_dir or script_dir / "wms_final_adapter"
    merged_model_dir = args.merged_model_dir or script_dir / "wms_final_model"

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_model_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
    sft_config = build_sft_config(SFTConfig, args=args, output_dir=output_dir)

    trainer = build_sft_trainer(
        SFTTrainer,
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_config,
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if not args.skip_merge:
        model.save_pretrained_merged(
            str(merged_model_dir),
            tokenizer,
            save_method="merged_16bit",
        )


def ensure_nvidia_library_path() -> None:
    if os.environ.get("WMS_NVIDIA_LIB_PATH_READY"):
        return

    nvidia_lib_dirs = find_nvidia_library_dirs()
    if not nvidia_lib_dirs:
        os.environ["WMS_NVIDIA_LIB_PATH_READY"] = "1"
        return

    current_dirs = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    missing_dirs = [path for path in nvidia_lib_dirs if path not in current_dirs]
    if not missing_dirs:
        os.environ["WMS_NVIDIA_LIB_PATH_READY"] = "1"
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(missing_dirs + current_dirs)
    env["WMS_NVIDIA_LIB_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def find_nvidia_library_dirs() -> list[str]:
    site_dirs = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(Path(user_site))

    nvidia_lib_dirs: list[str] = []
    for site_dir in site_dirs:
        nvidia_dir = site_dir / "nvidia"
        if not nvidia_dir.exists():
            continue
        nvidia_lib_dirs.extend(str(path) for path in sorted(nvidia_dir.glob("*/lib")) if path.is_dir())
    return nvidia_lib_dirs


def build_sft_config(sft_config_cls: type[Any], *, args: argparse.Namespace, output_dir: Path) -> Any:
    sft_config_kwargs = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "fp16": args.fp16,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "output_dir": str(output_dir),
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "report_to": "none",
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
    }
    return sft_config_cls(
        **resolve_supported_kwargs(
            sft_config_cls,
            sft_config_kwargs,
            aliases={
                "eval_strategy": "evaluation_strategy",
                "max_length": "max_seq_length",
            },
        )
    )


def build_sft_trainer(sft_trainer_cls: type[Any], **kwargs: Any) -> Any:
    return sft_trainer_cls(
        **resolve_supported_kwargs(
            sft_trainer_cls,
            kwargs,
            aliases={"processing_class": "tokenizer"},
        )
    )


def resolve_supported_kwargs(
    callable_obj: Any,
    kwargs: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs

    aliases = aliases or {}
    supported = set(signature.parameters)
    resolved: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in supported:
            resolved[key] = value
            continue
        alias = aliases.get(key)
        if alias and alias in supported:
            resolved[alias] = value
    return resolved


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune the WMS query-template extractor.")
    parser.add_argument("--data-path", type=Path, default=script_dir / "data" / "wms_data_enriched.jsonl")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "wms_checkpoints")
    parser.add_argument("--adapter-dir", type=Path, default=script_dir / "wms_final_adapter")
    parser.add_argument("--merged-model-dir", type=Path, default=script_dir / "wms_final_model")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-merge", action="store_true")
    return parser.parse_args()


def formatting_prompts_func(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
    instructions = examples.get("instruction", [])
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples.get("output", [])
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        question = build_question(instruction=instruction, input_text=input_text)
        template = build_training_template(question=question, output=output)
        prompt = build_query_template_prompt(question=question)
        texts.append(f"{prompt}{json.dumps(template, ensure_ascii=False, sort_keys=True)}")
    return {"text": texts}


def build_question(*, instruction: Any, input_text: Any) -> str:
    instruction_text = str(instruction or "").strip()
    input_value = str(input_text or "").strip()
    if input_value:
        return f"{instruction_text}\nContext: {input_value}"
    return instruction_text


def build_training_template(*, question: str, output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return normalize_template(output)

    output_text = str(output or "").strip()
    try:
        payload = json.loads(output_text)
        if isinstance(payload, dict):
            return normalize_template(payload)
    except json.JSONDecodeError:
        pass

    sql = extract_sql(output_text)
    return infer_template_from_sql(question=question, sql=sql)


def normalize_template(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), list) else []
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    sql = payload.get("sql")
    return {
        "intent": str(payload.get("intent") or "unknown"),
        "target": str(payload.get("target") or "unknown"),
        "filters": filters,
        "metrics": [str(metric) for metric in metrics],
        "limit": payload.get("limit") if isinstance(payload.get("limit"), int) else None,
        "sql": str(sql).strip() if isinstance(sql, str) and sql.strip() else None,
    }


def extract_sql(output_text: str) -> str:
    fenced = re.search(r"```sql\s*(.*?)```", output_text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    generic = re.search(r"```\s*(.*?)```", output_text, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return output_text.strip()


def infer_template_from_sql(*, question: str, sql: str) -> dict[str, Any]:
    lower_question = question.lower()
    lower_sql = sql.lower()
    target = infer_target(lower_question=lower_question, lower_sql=lower_sql)
    return {
        "intent": infer_intent(target=target, lower_question=lower_question, lower_sql=lower_sql),
        "target": target,
        "filters": infer_filters(sql=sql),
        "metrics": infer_metrics(lower_question=lower_question, lower_sql=lower_sql),
        "limit": infer_limit(lower_sql=lower_sql),
        "sql": sql or None,
    }


def infer_target(*, lower_question: str, lower_sql: str) -> str:
    table_targets = {
        "inventory": "inventory",
        "warehouse_inventory": "inventory",
        "movement_ledger": "inventory",
        "documents": "documents",
        "document_items": "documents",
        "positions": "positions",
        "warehouses": "warehouses",
        "products": "products",
        "customers": "customers",
        "orders": "orders",
        "reporting": "reporting",
    }
    for table, target in table_targets.items():
        if table in lower_sql or table in lower_question:
            return target
    return "unknown"


def infer_intent(*, target: str, lower_question: str, lower_sql: str) -> str:
    if any(word in lower_sql for word in ("sum(", "count(", "avg(", "group by")):
        return "report_lookup"
    if target == "inventory":
        return "inventory_lookup"
    if target == "documents":
        return "document_lookup"
    if target in {"warehouses", "positions"}:
        return "warehouse_lookup"
    if target == "products":
        return "product_lookup"
    if target == "customers":
        return "customer_lookup"
    if "order" in lower_question:
        return "order_status"
    return "unknown"


def infer_filters(*, sql: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    where_match = re.search(
        r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not where_match:
        return filters
    for key, value in re.findall(r"([a-zA-Z_][\w.]*?)\s*=\s*'([^']+)'", where_match.group(1)):
        filters[key.split(".")[-1]] = value
    for key, value in re.findall(r"([a-zA-Z_][\w.]*?)\s*=\s*(\d+)", where_match.group(1)):
        filters.setdefault(key.split(".")[-1], value)
    return filters


def infer_metrics(*, lower_question: str, lower_sql: str) -> list[str]:
    metrics = []
    metric_terms = {
        "quantity": ("quantity", "số lượng", "ton kho", "tồn kho"),
        "count": ("count(", "đếm", "bao nhiêu"),
        "sum": ("sum(", "tổng"),
        "average": ("avg(", "trung bình"),
        "amount": ("amount", "doanh thu", "giá trị"),
        "location": ("location", "vị trí"),
    }
    haystack = f"{lower_question} {lower_sql}"
    for metric, terms in metric_terms.items():
        if any(term in haystack for term in terms):
            metrics.append(metric)
    return metrics


def infer_limit(*, lower_sql: str) -> int | None:
    match = re.search(r"\blimit\s+(\d+)", lower_sql)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    main()
