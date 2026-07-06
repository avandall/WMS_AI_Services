from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    query: str
    mode: str
    documents: tuple[str, ...] = ()


class RetrievalPipeline:
    """Boundary for future retrieval/index lookups fed by AI-owned projections."""

    def build_context(self, *, question: str, mode: str) -> RetrievalContext:
        return RetrievalContext(query=question, mode=mode)
