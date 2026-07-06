from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared_utils.events import EventEnvelope


SNAPSHOT_EVENT_TYPES = {
    "DocumentProjectionSnapshot",
    "InventoryProjectionSnapshot",
    "ReportingProjectionSnapshot",
    "WarehouseProjectionSnapshot",
}

DOMAIN_EVENT_TYPES = {
    "DocumentUploaded",
    "DocumentPosted",
    "DocumentCancelled",
    "InventoryAdjusted",
    "InventoryMovementApplied",
    "StockReserved",
    "ReservationReleased",
}


@dataclass(frozen=True, slots=True)
class ReindexJob:
    job_id: str
    source_event_id: str
    source_event_type: str
    stream_id: str
    source_service: str
    occurred_at: str
    source_kind: str
    entity_type: str | None
    entity_id: str | None
    replay_of_event_id: str | None
    payload: dict[str, Any]


class EventIngestor:
    """Converts replayable event/projection envelopes into AI-owned reindex jobs."""

    def to_reindex_job(self, *, message_id: str, envelope: EventEnvelope) -> ReindexJob:
        source_kind = self._source_kind(envelope)
        payload = dict(envelope.payload)
        return ReindexJob(
            job_id=f"{envelope.event_id}:{message_id}",
            source_event_id=envelope.event_id,
            source_event_type=envelope.type,
            stream_id=message_id,
            source_service=envelope.source,
            occurred_at=envelope.occurred_at,
            source_kind=source_kind,
            entity_type=self._entity_type(envelope),
            entity_id=self._entity_id(payload),
            replay_of_event_id=self._replay_of_event_id(payload),
            payload=payload,
        )

    @staticmethod
    def _source_kind(envelope: EventEnvelope) -> str:
        if envelope.type in SNAPSHOT_EVENT_TYPES or envelope.payload.get("snapshot_type"):
            return "projection_snapshot"
        if envelope.type in DOMAIN_EVENT_TYPES or envelope.payload.get("entity_type"):
            return "domain_event"
        return "event_envelope"

    @staticmethod
    def _entity_type(envelope: EventEnvelope) -> str | None:
        payload = envelope.payload
        if payload.get("entity_type"):
            return str(payload["entity_type"])
        if "document_id" in payload:
            return "document"
        if "warehouse_id" in payload:
            return "warehouse"
        if "product_id" in payload:
            return "product"
        return None

    @staticmethod
    def _entity_id(payload: dict[str, Any]) -> str | None:
        for key in ("entity_id", "document_id", "warehouse_id", "product_id", "customer_id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _replay_of_event_id(payload: dict[str, Any]) -> str | None:
        value = payload.get("replay_of_event_id")
        return str(value) if value is not None else None
