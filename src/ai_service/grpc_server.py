from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
from shared_utils.observability import grpc_observability_interceptor
from shared_utils.security import add_configured_grpc_port

from ai_service.event_consumer import start_ai_reindex_consumer_thread
from ai_service.grpc_servicer import AIServiceServicer
from ai_service.grpc_servicer import add_AIServiceServicer_to_server


def _prewarm(servicer: AIServiceServicer) -> None:
    try:
        pipeline = servicer._get_pipeline()
        # Force the heavy engine init (downloads HF embedding model) so the
        # first real query doesn't block waiting for it.
        pipeline.provider._get_engine()
        print("AI pipeline pre-warm complete.", file=sys.stderr)
    except Exception as exc:
        print(f"AI pipeline pre-warm failed (non-fatal): {exc}", file=sys.stderr)


def serve(*, host: str = "0.0.0.0", port: int = 50059) -> None:
    start_ai_reindex_consumer_thread()
    servicer = AIServiceServicer()
    # Pre-warm in background so gRPC server starts immediately
    threading.Thread(target=_prewarm, args=(servicer,), daemon=True).start()
    server = grpc.server(
        ThreadPoolExecutor(max_workers=10),
        interceptors=[grpc_observability_interceptor(service="ai-service")],
    )
    add_AIServiceServicer_to_server(servicer, server)
    add_configured_grpc_port(server, host, port)
    server.start()
    server.wait_for_termination()
