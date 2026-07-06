from __future__ import annotations

import uvicorn

from ai_service.app import create_app
from ai_service.grpc_server import serve


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="info")


if __name__ == "__main__":
    main()


def main_grpc() -> None:
    serve(host="0.0.0.0", port=50059)

