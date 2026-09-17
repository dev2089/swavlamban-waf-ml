"""Compatibility entry point for the canonical production FastAPI application.

The active WAF implementation lives under ``waf/``. This module deliberately
contains no legacy demo routes, hard-coded metrics, or separate security path.
"""
from __future__ import annotations

import os

import uvicorn

from waf.api.production_api import app

__all__ = ["app"]


if __name__ == "__main__":
    uvicorn.run(
        "waf.api.production_api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
