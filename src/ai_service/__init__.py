from __future__ import annotations

import sys
from pathlib import Path

_gen_path = Path(__file__).resolve().parent / "gen"
if str(_gen_path) not in sys.path:
    sys.path.insert(0, str(_gen_path))

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from ai_service.app import create_app

        return create_app
    raise AttributeError(name)
