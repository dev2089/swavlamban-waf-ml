from __future__ import annotations

import math
import re

from waf.core.models import FeatureVector, RequestEnvelope


_TOKEN_PATTERNS = {
    "has_sql_keyword": re.compile(r"\b(select|union|insert|update|delete|drop)\b", re.I),
    "has_xss_token": re.compile(r"<\s*script\b|javascript\s*:", re.I),
    "has_traversal": re.compile(r"(?:\.\./|\.\.\\)", re.I),
    "has_command_token": re.compile(r"(?:;|&&|\|\||`|\$\(|\bcurl\b|\bwget\b)", re.I),
}


class HTTPFeatureExtractor:
    """Deterministic baseline HTTP feature extractor."""

    schema_version = "http-v1"

    def extract(self, request: RequestEnvelope) -> FeatureVector:
        target = f"{request.path}?{request.query}" if request.query else request.path
        body_text = request.body.decode("utf-8", errors="replace")
        combined = target + "\n" + body_text
        entropy = self._entropy(combined.encode("utf-8"))
        return FeatureVector(
            schema_version=self.schema_version,
            values={
                "method_code": float(self._method_code(request.method)) / 7.0,
                "path_length": min(len(request.path), 4096) / 4096.0,
                "query_length": min(len(request.query), 8192) / 8192.0,
                "body_length": min(len(request.body), 1_048_576) / 1_048_576.0,
                "header_count": min(len(request.headers), 128) / 128.0,
                "target_entropy": min(entropy, 8.0) / 8.0,
                "has_sql_keyword": float(bool(_TOKEN_PATTERNS["has_sql_keyword"].search(combined))),
                "has_xss_token": float(bool(_TOKEN_PATTERNS["has_xss_token"].search(combined))),
                "has_traversal": float(bool(_TOKEN_PATTERNS["has_traversal"].search(combined))),
                "has_command_token": float(bool(_TOKEN_PATTERNS["has_command_token"].search(combined))),
            },
        )

    @staticmethod
    def _method_code(method: str) -> int:
        return {
            "GET": 1,
            "HEAD": 2,
            "POST": 3,
            "PUT": 4,
            "PATCH": 5,
            "DELETE": 6,
            "OPTIONS": 7,
        }.get(method.upper(), 0)

    @staticmethod
    def _entropy(data: bytes) -> float:
        if not data:
            return 0.0
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        size = len(data)
        return -sum((n / size) * math.log2(n / size) for n in counts if n)
