from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WAFConfig:
    """Non-secret runtime policy for the WAF edge; Phase 5 callers opt into the phase5 pipeline version."""

    pipeline_version: str = "phase3"
    block_threshold: float = 0.80
    alert_threshold: float = 0.50
    max_body_bytes: int = 1_048_576
    feature_schema_version: str = "http-v2"
    upstream_url: str = "http://127.0.0.1:9000"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    request_timeout_seconds: float = 10.0
    max_response_bytes: int = 10_485_760

    @classmethod
    def from_env(cls) -> "WAFConfig":
        def bounded_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            value = default if raw is None else float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            return value
        max_body = int(os.getenv("WAF_MAX_BODY_BYTES", "1048576"))
        if max_body <= 0:
            raise ValueError("WAF_MAX_BODY_BYTES must be positive")
        max_response = int(os.getenv("WAF_MAX_RESPONSE_BYTES", "10485760"))
        if max_response <= 0:
            raise ValueError("WAF_MAX_RESPONSE_BYTES must be positive")
        listen_port = int(os.getenv("WAF_LISTEN_PORT", "8080"))
        if not 1 <= listen_port <= 65535:
            raise ValueError("WAF_LISTEN_PORT must be in [1, 65535]")
        timeout = float(os.getenv("WAF_REQUEST_TIMEOUT_SECONDS", "10.0"))
        if timeout <= 0:
            raise ValueError("WAF_REQUEST_TIMEOUT_SECONDS must be positive")
        return cls(
            pipeline_version=os.getenv("WAF_PIPELINE_VERSION", "phase3"),
            block_threshold=bounded_float("WAF_BLOCK_THRESHOLD", 0.80),
            alert_threshold=bounded_float("WAF_ALERT_THRESHOLD", 0.50),
            max_body_bytes=max_body,
            feature_schema_version=os.getenv("WAF_FEATURE_SCHEMA", "http-v2"),
            upstream_url=os.getenv("WAF_UPSTREAM_URL", "http://127.0.0.1:9000"),
            listen_host=os.getenv("WAF_LISTEN_HOST", "127.0.0.1"),
            listen_port=listen_port,
            request_timeout_seconds=timeout,
            max_response_bytes=max_response,
        )
