from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ai_service.pipeline.ingestion import ReindexJob


class ReindexJobStore(Protocol):
    def enqueue(self, job: ReindexJob) -> None: ...


class JsonlReindexJobStore:
    """Small local queue adapter used until an AI-owned durable queue/table is required."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def enqueue(self, job: ReindexJob) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(job), ensure_ascii=False, sort_keys=True) + "\n")
