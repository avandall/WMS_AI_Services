from __future__ import annotations

import json

import grpc

from ai_service.gen.wms.ai.v1 import ai_pb2, ai_pb2_grpc
from ai_service.pipeline import AIQueryPipeline, WMSEngineProviderAdapter


class AIServiceServicer(ai_pb2_grpc.AIServiceServicer):
    _pipeline: AIQueryPipeline | None = None

    @classmethod
    def _get_pipeline(cls) -> AIQueryPipeline:
        if cls._pipeline is None:
            cls._pipeline = AIQueryPipeline(provider=WMSEngineProviderAdapter())
        return cls._pipeline

    @classmethod
    def reset_pipeline(cls) -> None:
        cls._pipeline = None

    def Query(self, request: ai_pb2.QueryRequest, context: grpc.ServicerContext):
        # Read request id for observability (future structured logs)
        for k, v in context.invocation_metadata() or []:
            if k.lower() == "x-request-id":
                break
        mode = (request.mode or "auto").lower()
        try:
            result = self._get_pipeline().answer(question=request.question, mode=mode)
            return ai_pb2.QueryResponse(
                success=result.success,
                mode=result.mode,
                response=result.response,
                error=result.error,
            )
        except Exception as exc:
            return ai_pb2.QueryResponse(success=False, mode=mode, response="", error=str(exc))

    def Status(self, request: ai_pb2.StatusRequest, context: grpc.ServicerContext):
        try:
            info = self._get_pipeline().status()
            return ai_pb2.StatusResponse(json=json.dumps(info, ensure_ascii=False, default=str))
        except Exception as exc:
            return ai_pb2.StatusResponse(json=json.dumps({"error": str(exc)}, ensure_ascii=False))


add_AIServiceServicer_to_server = ai_pb2_grpc.add_AIServiceServicer_to_server
