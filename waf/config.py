from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Settings:
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    max_body_bytes: int = 1_048_576
    telemetry_queue_max: int = 10_000
    @classmethod
    def from_env(cls) -> "Settings":
        port = int(os.getenv("WAF_PORT", str(cls.port)))
        max_body = int(os.getenv("WAF_MAX_BODY_BYTES", str(cls.max_body_bytes)))
        queue_max = int(os.getenv("WAF_TELEMETRY_QUEUE_MAX", str(cls.telemetry_queue_max)))
        if not 1 <= port <= 65535: raise ValueError("WAF_PORT must be between 1 and 65535")
        if max_body <= 0: raise ValueError("WAF_MAX_BODY_BYTES must be positive")
        if queue_max <= 0: raise ValueError("WAF_TELEMETRY_QUEUE_MAX must be positive")
        return cls(environment=os.getenv("WAF_ENV", cls.environment), host=os.getenv("WAF_HOST", cls.host), port=port, log_level=os.getenv("WAF_LOG_LEVEL", cls.log_level).upper(), max_body_bytes=max_body, telemetry_queue_max=queue_max)