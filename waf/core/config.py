from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WAFConfig:
    """Non-secret runtime policy for the WAF edge."""

    pipeline_version: str = "phase2"
    block_threshold: float = 0.80
    alert_threshold: float = 0.50
    max_body_bytes: int = 1_048_576
    feature_schema_version: str = "http-v1"
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
        return cls(
            pipeline_version=os.getenv("WAF_PIPELINE_VERSION", cls.pipeline_version),
            block_threshold=bounded_float("WAF_BLOCK_THRESHOLD", cls.block_threshold),
            alert_threshold=bounded_float("WAF_ALERT_THRESHOLD", cls.alert_threshold),
            max_body_bytes=max_body,
            feature_schema_version=os.getenv("WAF_FEATURE_SCHEMA", cls.feature_schema_version),
            upstream_url=os.getenv("WAF_UPSTREAM_URL", cls.upstream_url),
            listen_host=os.getenv("WAF_LISTEN_HOST", cls.listen_host),
            listen_port=int(os.getenv("WAF_LISTEN_PORT", str(cls.listen_port))),
            request_timeout_seconds=float(os.getenv("WAF_REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))),
            max_response_bytes=max_response,
        )
