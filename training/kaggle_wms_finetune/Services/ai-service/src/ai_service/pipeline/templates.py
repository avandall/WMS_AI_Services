from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class QueryTemplate:
    intent: str
    target: str
    filters: dict[str, Any]
    metrics: tuple[str, ...] = ()
    limit: int | None = None
    sql: str | None = None
    raw_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryTemplateExtractor(Protocol):
    def extract(self, *, question: str) -> QueryTemplate: ...


QUERY_TEMPLATE_SCHEMA = """{
  "intent": "inventory_lookup|order_status|report_lookup|warehouse_lookup|document_lookup|product_lookup|customer_lookup|unknown",
  "target": "inventory|orders|reporting|warehouses|positions|documents|products|customers|unknown",
  "filters": {"key": "value"},
  "metrics": ["quantity"],
  "limit": 20,
  "sql": "optional SQL query when the request needs a database query"
}"""


def build_query_template_prompt(*, question: str) -> str:
    return f"""You are a WMS query planner.
Return only valid JSON with this shape:
{QUERY_TEMPLATE_SCHEMA}

Question: {question}
"""


class GroqQueryTemplateExtractor:
    """
    Default template extractor backed by Groq.

    This is intentionally behind a small object boundary so it can be replaced
    with a local model later without changing AIQueryPipeline.
    """

    def __init__(self):
        from ai_engine.config import settings
        from langchain_groq import ChatGroq

        self.llm = ChatGroq(**settings.get_llm_config())

    def extract(self, *, question: str) -> QueryTemplate:
        from langchain_core.messages import HumanMessage

        prompt = build_query_template_prompt(question=question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return parse_query_template_content(question=question, content=response.content)


class FineTunedQueryTemplateExtractor:
    """
    Optional local extractor backed by a fine-tuned causal LM.

    It is used when FINE_TUNED_MODEL_PATH is configured. The model is prompted
    to return the same JSON template as the Groq extractor so the rest of the
    pipeline remains unchanged.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 256):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._generator = None

    def _get_generator(self):
        if self._generator is not None:
            return self._generator

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model_path = Path(self.model_path)
        if (model_path / "adapter_config.json").exists():
            from peft import AutoPeftModelForCausalLM

            model = AutoPeftModelForCausalLM.from_pretrained(self.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)

        if self.device.isdigit():
            device_index = int(self.device)
        elif self.device.lower() in {"cpu", "mps", "cuda", "auto"}:
            device_index = 0 if self.device.lower() == "cuda" else -1
        else:
            device_index = -1

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )
        return self._generator

    def extract(self, *, question: str) -> QueryTemplate:
        prompt = build_query_template_prompt(question=question)
        generator = self._get_generator()
        result = generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        content = result[0].get("generated_text", "") if result else ""
        return parse_query_template_content(question=question, content=content)


class HeuristicQueryTemplateExtractor:
    """Fallback extractor used when the Groq extractor cannot produce a template."""

    _sku_pattern = re.compile(r"\b([A-Z]{2,}-\d{2,})\b")

    def extract(self, *, question: str) -> QueryTemplate:
        sku = self._sku_pattern.search(question)
        filters: dict[str, Any] = {}
        if sku:
            filters["sku"] = sku.group(1)
        return QueryTemplate(
            intent="inventory_lookup" if filters else "unknown",
            target="inventory" if filters else "unknown",
            filters=filters,
            metrics=("quantity", "location") if filters else (),
            raw_question=question,
        )


class SafeQueryTemplateExtractor:
    def __init__(self, primary: QueryTemplateExtractor | None = None, fallback: QueryTemplateExtractor | None = None):
        self.primary = primary or _default_primary_extractor()
        self.fallback = fallback or HeuristicQueryTemplateExtractor()

    def extract(self, *, question: str) -> QueryTemplate:
        try:
            template = self.primary.extract(question=question)
            if template.intent != "unknown" or template.filters:
                return template
        except Exception:
            pass
        return self.fallback.extract(question=question)


def parse_query_template_content(*, question: str, content: str) -> QueryTemplate:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    return _template_from_payload(question=question, payload=payload)


def _template_from_payload(*, question: str, payload: dict[str, Any]) -> QueryTemplate:
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), list) else []
    limit = payload.get("limit")
    sql = payload.get("sql")
    return QueryTemplate(
        intent=str(payload.get("intent") or "unknown"),
        target=str(payload.get("target") or "unknown"),
        filters=filters,
        metrics=tuple(str(metric) for metric in metrics),
        limit=int(limit) if isinstance(limit, int) else None,
        sql=str(sql).strip() if isinstance(sql, str) and sql.strip() else None,
        raw_question=question,
    )


def _default_primary_extractor() -> QueryTemplateExtractor:
    from ai_engine.config import settings
    from ai_engine.utils import logger

    if settings.FINE_TUNED_MODEL_PATH:
        try:
            return FineTunedQueryTemplateExtractor(
                model_path=settings.FINE_TUNED_MODEL_PATH,
                device=settings.FINE_TUNED_MODEL_DEVICE,
                max_new_tokens=settings.FINE_TUNED_MAX_NEW_TOKENS,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to Groq extractor because fine-tuned model could not be loaded",
                extra={"model_path": settings.FINE_TUNED_MODEL_PATH, "error": str(exc)},
            )
    return GroqQueryTemplateExtractor()


def extractor_source(extractor: QueryTemplateExtractor) -> str:
    if isinstance(extractor, FineTunedQueryTemplateExtractor):
        return "fine_tuned"
    if isinstance(extractor, GroqQueryTemplateExtractor):
        return "groq"
    if isinstance(extractor, HeuristicQueryTemplateExtractor):
        return "heuristic"
    if isinstance(extractor, SafeQueryTemplateExtractor):
        return extractor_source(extractor.primary)
    return extractor.__class__.__name__
