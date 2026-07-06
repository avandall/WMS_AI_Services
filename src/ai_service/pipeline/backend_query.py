from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ai_service.pipeline.templates import QueryTemplate


@dataclass(frozen=True, slots=True)
class BackendQueryResponse:
    success: bool
    payload: dict[str, Any]
    error: str = ""


class BackendQueryClient(Protocol):
    def execute(self, *, template: QueryTemplate) -> BackendQueryResponse: ...


class TemplateBackendQueryClient:
    """
    Backend query boundary for structured data requests.

    It accepts normalized key-value templates and returns a structured payload.
    A real gRPC/HTTP backend adapter can replace this class without changing the
    router or template extractor.
    """

    def execute(self, *, template: QueryTemplate) -> BackendQueryResponse:
        answer = _template_to_natural_language(template)
        return BackendQueryResponse(
            success=True,
            payload={"answer": answer},
        )


class HttpBackendQueryClient:
    """POSTs query templates to a backend endpoint that owns operational data access."""

    def __init__(self, endpoint: str, auth_token: str | None = None, timeout_seconds: float = 10.0):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    def execute(self, *, template: QueryTemplate) -> BackendQueryResponse:
        body = json.dumps({"template": template.to_dict()}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8") or "{}")
            return BackendQueryResponse(success=True, payload=payload)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return BackendQueryResponse(success=False, payload={}, error=str(exc))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers


def default_backend_query_client() -> BackendQueryClient:
    endpoint = os.getenv("AI_BACKEND_QUERY_URL", "").strip()
    if endpoint:
        return HttpBackendQueryClient(
            endpoint=endpoint,
            auth_token=os.getenv("AI_BACKEND_QUERY_TOKEN") or None,
            timeout_seconds=float(os.getenv("AI_BACKEND_QUERY_TIMEOUT", "10")),
        )
    return TemplateBackendQueryClient()


def render_backend_response(response: BackendQueryResponse) -> str:
    if "answer" in response.payload:
        return str(response.payload["answer"])
    return json.dumps(response.payload, ensure_ascii=False, sort_keys=True)


def _template_to_natural_language(template: QueryTemplate) -> str:
    """Convert a parsed query template into a human-readable response."""
    intent = template.intent
    target = template.target
    q = template.raw_question.strip()

    if intent == "unknown" and target == "unknown":
        return (
            "I'm a WMS data assistant. I can help you look up inventory levels, "
            "warehouse stock, products, documents, customers, and sales reports. "
            "Try asking something like: 'How many units of product X are in warehouse Y?' "
            "or 'Show me all sale documents this month'."
        )

    filters_desc = ""
    if template.filters:
        parts = [f"{k}={v}" for k, v in template.filters.items()]
        filters_desc = " with filters: " + ", ".join(parts)

    metrics_desc = ""
    if template.metrics:
        metrics_desc = " — looking for: " + ", ".join(template.metrics)

    target_labels = {
        "inventory": "inventory",
        "orders": "orders",
        "reporting": "reports",
        "warehouses": "warehouses",
        "documents": "documents",
        "products": "products",
        "customers": "customers",
        "positions": "stock positions",
    }
    target_label = target_labels.get(target, target)

    intent_labels = {
        "inventory_lookup": "Inventory lookup",
        "order_status": "Order status",
        "report_lookup": "Report",
        "warehouse_lookup": "Warehouse lookup",
        "document_lookup": "Document lookup",
        "product_lookup": "Product lookup",
        "customer_lookup": "Customer lookup",
    }
    intent_label = intent_labels.get(intent, intent.replace("_", " ").capitalize())

    limit_desc = f" (top {template.limit})" if template.limit else ""

    return (
        f"{intent_label} for {target_label}{filters_desc}{metrics_desc}{limit_desc}. "
        f"To get live data, please configure AI_BACKEND_QUERY_URL in the ai-service "
        f"environment pointing to the API gateway (e.g. http://api-gateway:8000/api/v1/ai/backend-query)."
    )
