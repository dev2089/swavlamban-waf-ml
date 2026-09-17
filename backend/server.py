"""Compatibility entry point for the secure Phase 9 FastAPI service."""
from __future__ import annotations

import os
import uvicorn

from waf.api.production_api import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "backend.server:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
