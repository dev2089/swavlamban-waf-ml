from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WAFConfig:
    """Non-secret runtime policy for the pipeline."""

    pipeline_version: str = "phase1"
    block_threshold: float = 0.80
    alert_threshold: float = 0.50
    max_body_bytes: int = 1_048_576
    feature_schema_version: str = "http-v1"

    @classmethod
    def from_env(cls) -> "WAFConfig":
        def bounded_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            value = default if raw is None else float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            return value

        max_body = int(os.getenv("WAF_MAX_BODY_BYTES", str(cls.max_body_bytes)))
        if max_body <= 0:
            raise ValueError("WAF_MAX_BODY_BYTES must be positive")
        return cls(
            pipeline_version=os.getenv("WAF_PIPELINE_VERSION", cls.pipeline_version),
            block_threshold=bounded_float("WAF_BLOCK_THRESHOLD", cls.block_threshold),
            alert_threshold=bounded_float("WAF_ALERT_THRESHOLD", cls.alert_threshold),
            max_body_bytes=max_body,
            feature_schema_version=os.getenv("WAF_FEATURE_SCHEMA", cls.feature_schema_version),
        )
