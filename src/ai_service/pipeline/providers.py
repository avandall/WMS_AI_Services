from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ai_service.pipeline.generation import QueryResult


class AIProvider(Protocol):
    def generate(self, *, question: str, mode: str) -> "QueryResult": ...

    def status(self) -> dict[str, object]: ...


class WMSEngineProviderAdapter:
    """Adapter around the heavy RAG engine; imports stay inside the opt-in AI runtime."""

    _engine: object | None = None

    def _get_engine(self):
        if self._engine is None:
            from ai_engine.core.engine import ProcessingMode, WMSEngine

            self._engine = WMSEngine(mode=ProcessingMode.RAG)
        return self._engine

    def generate(self, *, question: str, mode: str) -> "QueryResult":
        from ai_engine.core.engine import ProcessingMode
        from ai_service.pipeline.generation import QueryResult

        selected_mode = mode if mode in {"rag", "agent", "hybrid"} else "rag"
        coro = self._get_engine().process_query(
            question,
            mode=ProcessingMode(selected_mode),
        )
        result = asyncio.run(coro) if asyncio.iscoroutine(coro) else coro
        return QueryResult(
            success=bool(result.get("success", False)),
            mode=str(result.get("mode", selected_mode)),
            response=str(result.get("response", "")),
            error=str(result.get("error", "")),
        )

    def status(self) -> dict[str, object]:
        return dict(self._get_engine().get_engine_info())
