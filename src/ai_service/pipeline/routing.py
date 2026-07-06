from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    confidence: float
    reason: str


class QueryRouter(Protocol):
    def route(self, *, question: str, requested_mode: str) -> RouteDecision: ...


class HeuristicQueryRouter:
    """Small deterministic router that keeps SQL/data requests out of the RAG path."""

    _data_patterns = (
        re.compile(r"\b(sql|select|query|redis|database|db|table|record)\b", re.IGNORECASE),
        re.compile(r"\b(how many|where is|stock|inventory|quantity|sku|order|status)\b", re.IGNORECASE),
        re.compile(r"\b[A-Z]{2,}-\d{2,}\b"),
    )

    def route(self, *, question: str, requested_mode: str) -> RouteDecision:
        mode = requested_mode.lower().strip()
        if mode in {"rag", "data_query", "backend_query"}:
            return RouteDecision(route="data_query" if mode != "rag" else "rag", confidence=1.0, reason="explicit mode")
        if mode in {"agent", "hybrid", "auto", ""}:
            if any(pattern.search(question) for pattern in self._data_patterns):
                return RouteDecision(route="data_query", confidence=0.75, reason="data-query keywords detected")
            # Default to data_query for auto mode — RAG path is too slow without warm LLM
            return RouteDecision(route="data_query", confidence=0.6, reason="default to data_query for WMS context")
        return RouteDecision(route="data_query", confidence=0.5, reason="unknown mode fallback")
