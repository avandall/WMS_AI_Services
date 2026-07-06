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
    return f"""You are a WMS query planner. Extract structured query parameters from the user question.
Return ONLY valid JSON with this exact shape (no explanation, no markdown):
{QUERY_TEMPLATE_SCHEMA}

Rules:
- If the question mentions a product ID or product number, put it in filters as "product_id"
- If the question mentions a warehouse ID or warehouse number, put it in filters as "warehouse_id"
- If the question asks about quantity/stock/inventory (including physical quantity, reserved stock, incoming stock, in-transit stock, or available quantity), set intent="inventory_lookup" and target="inventory"
- If the question asks about customers, set intent="customer_lookup" and target="customers"
- If the question asks about documents/orders/sales or pending execution, set intent="document_lookup" and target="documents"
- If the question asks about transaction types or the stock ledger, set intent="document_lookup" and target="documents"
- If the question asks about products, set intent="product_lookup" and target="products"
- If the question asks about warehouses, set intent="warehouse_lookup" and target="warehouses"
- Always extract numeric IDs from the question into filters
- Set limit to a reasonable number (default 20)

Examples:
Q: "how many product ID 2 in warehouse ID 2"
A: {{"intent":"inventory_lookup","target":"inventory","filters":{{"product_id":"2","warehouse_id":"2"}},"metrics":["quantity"],"limit":1,"sql":null}}

Q: "show reserved stock of product 5"
A: {{"intent":"inventory_lookup","target":"inventory","filters":{{"product_id":"5"}},"metrics":["reserved_qty"],"limit":20,"sql":null}}

Q: "list pending execution documents"
A: {{"intent":"document_lookup","target":"documents","filters":{{"status":"pending_execution"}},"metrics":["status"],"limit":20,"sql":null}}

Q: "show stock ledger for warehouse 1"
A: {{"intent":"document_lookup","target":"documents","filters":{{"warehouse_id":"1"}},"metrics":["transaction_type"],"limit":20,"sql":null}}

Q: "available quantity of product 3"
A: {{"intent":"inventory_lookup","target":"inventory","filters":{{"product_id":"3"}},"metrics":["available_qty"],"limit":20,"sql":null}}

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
    _product_id_pattern = re.compile(r"product\s+(?:id\s+|ID\s+|#\s*)?(\d+)", re.IGNORECASE)
    _warehouse_id_pattern = re.compile(r"warehouse\s+(?:id\s+|ID\s+|#\s*)?(\d+)", re.IGNORECASE)
    _inventory_keywords = re.compile(r"\b(how many|quantity|stock|inventory|units?|items? in)\b", re.IGNORECASE)
    _customer_keywords = re.compile(r"\b(list (all )?customers|show (all )?customers|customer (id|name|debt|list))\b", re.IGNORECASE)
    _product_keywords = re.compile(r"\b(list (all )?products|show (all )?products|product (id|name|price|list))\b", re.IGNORECASE)
    _document_keywords = re.compile(r"\b(list (all )?(documents?|orders?|sales?|invoices?)|show (all )?(documents?|sale documents?))\b", re.IGNORECASE)
    _warehouse_list_keywords = re.compile(r"\b(list (all )?warehouses?|show (all )?warehouses?|warehouse (id|name|list))\b", re.IGNORECASE)

    # Phase 15: New keywords for WMS terms
    _available_keywords = re.compile(r"\b(available quantity|available stock|available|atp)\b", re.IGNORECASE)
    _reserved_keywords = re.compile(r"\b(reserved stock|reserved quantity|reserved|reservation)\b", re.IGNORECASE)
    _in_transit_keywords = re.compile(r"\b(in-transit|in transit|transit stock|transfer issue|transfer receipt)\b", re.IGNORECASE)
    _transaction_type_keywords = re.compile(r"\b(transaction type|reason code)\b", re.IGNORECASE)
    _stock_ledger_keywords = re.compile(r"\b(stock ledger|ledger|movement ledger|inventory transaction)\b", re.IGNORECASE)
    _pending_execution_keywords = re.compile(r"\b(pending execution|unexecuted|execution status|in progress|draft|requested)\b", re.IGNORECASE)

    def extract(self, *, question: str) -> QueryTemplate:
        filters: dict[str, Any] = {}

        # Extract explicit numeric IDs first
        product_match = self._product_id_pattern.search(question)
        if product_match:
            filters["product_id"] = product_match.group(1)

        warehouse_match = self._warehouse_id_pattern.search(question)
        if warehouse_match:
            filters["warehouse_id"] = warehouse_match.group(1)

        sku = self._sku_pattern.search(question)
        if sku:
            filters["sku"] = sku.group(1)

        # Check new matrix/ledger terms first to avoid generic inventory fallback
        metrics = ["quantity"]
        is_matrix_query = False
        if self._available_keywords.search(question):
            metrics.append("available_qty")
            is_matrix_query = True
        if self._reserved_keywords.search(question):
            metrics.append("reserved_qty")
            is_matrix_query = True
        if self._in_transit_keywords.search(question):
            metrics.append("in_transit_qty")
            is_matrix_query = True

        if is_matrix_query:
            return QueryTemplate(
                intent="inventory_lookup",
                target="inventory",
                filters=filters,
                metrics=tuple(metrics),
                raw_question=question,
            )

        if self._transaction_type_keywords.search(question) or self._stock_ledger_keywords.search(question):
            return QueryTemplate(
                intent="document_lookup",
                target="documents",
                filters=filters,
                metrics=("transaction_type", "reason_code", "status"),
                raw_question=question,
            )

        if self._pending_execution_keywords.search(question):
            return QueryTemplate(
                intent="document_lookup",
                target="documents",
                filters=filters,
                metrics=("status", "execution_status"),
                raw_question=question,
            )

        # If any IDs were found, it's definitely an inventory lookup
        if filters:
            return QueryTemplate(
                intent="inventory_lookup",
                target="inventory",
                filters=filters,
                metrics=("quantity", "location"),
                raw_question=question,
            )

        # Match broader inventory keywords only (not just "inventory" the word)
        if self._inventory_keywords.search(question):
            return QueryTemplate(
                intent="inventory_lookup",
                target="inventory",
                filters=filters,
                metrics=("quantity",),
                raw_question=question,
            )

        if self._customer_keywords.search(question):
            return QueryTemplate(intent="customer_lookup", target="customers", filters={}, metrics=("name", "debt"), raw_question=question)

        if self._document_keywords.search(question):
            return QueryTemplate(intent="document_lookup", target="documents", filters={}, metrics=("status", "total"), raw_question=question)

        if self._warehouse_list_keywords.search(question):
            return QueryTemplate(intent="warehouse_lookup", target="warehouses", filters={}, metrics=("name",), raw_question=question)

        if self._product_keywords.search(question):
            return QueryTemplate(intent="product_lookup", target="products", filters={}, metrics=("name", "price"), raw_question=question)

        # Unknown — will fall through to Groq chat
        return QueryTemplate(intent="unknown", target="unknown", filters={}, metrics=(), raw_question=question)


class SafeQueryTemplateExtractor:
    def __init__(self, primary: QueryTemplateExtractor | None = None, fallback: QueryTemplateExtractor | None = None):
        self.primary = primary or _default_primary_extractor()
        self.fallback = fallback or HeuristicQueryTemplateExtractor()
        self._heuristic = HeuristicQueryTemplateExtractor()

    def extract(self, *, question: str) -> QueryTemplate:
        # Always run heuristic first to extract any numeric IDs from the question
        heuristic = self._heuristic.extract(question=question)

        try:
            template = self.primary.extract(question=question)
            # If Groq returned unknown but heuristic found something useful, merge
            if template.intent == "unknown" and heuristic.intent != "unknown":
                return heuristic
            # If Groq found intent but missed filters that heuristic caught, merge filters
            merged_filters = {**heuristic.filters, **template.filters}
            if merged_filters != template.filters:
                return QueryTemplate(
                    intent=template.intent,
                    target=template.target,
                    filters=merged_filters,
                    metrics=template.metrics or heuristic.metrics,
                    limit=template.limit,
                    sql=template.sql,
                    raw_question=question,
                )
            return template
        except Exception:
            pass
        return heuristic


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
