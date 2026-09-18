from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from urllib.parse import parse_qsl, unquote, unquote_plus

from waf.core.models import FeatureVector, RequestEnvelope

SCHEMA_VERSION = "http-v2"

_METHODS = {
    "GET": 1, "HEAD": 2, "POST": 3, "PUT": 4, "PATCH": 5,
    "DELETE": 6, "OPTIONS": 7, "CONNECT": 8, "TRACE": 9,
}
_SQL = re.compile(r"\b(select|union|insert|update|delete|drop|alter|create)\b", re.I)
_XSS = re.compile(r"<\s*script\b|javascript\s*:|on(?:error|load|click)\s*=", re.I)
_TRAVERSAL = re.compile(r"(?:\.\./|\.\.\\)", re.I)
_COMMAND = re.compile(r"(?:;\s*(?:cat|id|uname|curl|wget|bash|sh)\b|\$\(|\x60[^\x60]+\x60|&&|\|\|)", re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


class ProductionHTTPFeatureExtractor:
    """Deterministic, bounded HTTP feature pipeline for the WAF fast path."""

    schema_version = SCHEMA_VERSION
    max_headers = 128
    max_header_value_chars = 4096
    max_header_bytes = 16_384
    max_path_chars = 8192
    max_query_chars = 16_384
    max_query_fields = 256
    max_body_scan_bytes = 262_144

    def extract(self, request: RequestEnvelope) -> FeatureVector:
        path = self._bounded_text(request.path, self.max_path_chars)
        query = self._bounded_text(request.query, self.max_query_chars)
        body = request.body[: self.max_body_scan_bytes]

        normalized_target = normalize_target(path, query)
        body_text = self._decode_body(body)
        combined = (normalized_target + "\n" + body_text)[
            : self.max_query_chars + self.max_body_scan_bytes
        ]

        headers = self._normalized_headers(request.headers)
        query_pairs, query_parse_overflow = parse_query_safely(query)
        query_keys = [key for key, _ in query_pairs]
        content_type = headers.get("content-type", "").lower()
        header_bytes = sum(len(key) + len(value) for key, value in headers.items())
        header_value_chars = sum(len(value) for value in headers.values())

        values = {
            "method_code": _METHODS.get(request.method.upper(), 0) / 9.0,
            "method_known": float(request.method.upper() in _METHODS),
            "scheme_https": float(request.scheme.lower() == "https"),
            "host_length": min(len(request.host), 255) / 255.0,
            "path_length": min(len(path), self.max_path_chars) / self.max_path_chars,
            "normalized_path_length": min(
                len(normalized_target.split("?", 1)[0]), self.max_path_chars
            ) / self.max_path_chars,
            "query_length": min(len(query), self.max_query_chars) / self.max_query_chars,
            "body_length": min(len(request.body), 1_048_576) / 1_048_576.0,
            "header_count": min(len(headers), self.max_headers) / self.max_headers,
            "header_bytes": min(header_bytes, self.max_header_bytes) / self.max_header_bytes,
            "header_value_bytes": min(
                header_value_chars, self.max_header_bytes
            ) / self.max_header_bytes,
            "query_param_count": min(len(query_pairs), 128) / 128.0,
            "unique_query_key_count": min(len(set(query_keys)), 128) / 128.0,
            "duplicate_query_key_count": min(
                max(0, len(query_keys) - len(set(query_keys))), 32
            ) / 32.0,
            "query_parse_overflow": float(query_parse_overflow),
            "cookie_count": min(cookie_count(headers.get("cookie", "")), 64) / 64.0,
            "has_json_body": float("json" in content_type),
            "has_form_body": float("x-www-form-urlencoded" in content_type),
            "has_xml_body": float("xml" in content_type),
            "has_multipart_body": float("multipart/" in content_type),
            "has_content_length": float("content-length" in headers),
            "content_length_mismatch": content_length_mismatch(
                headers, len(request.body)
            ),
            "percent_encoded_ratio": len(_PERCENT.findall(path + query))
            / max(1, len(path + query)),
            "malformed_percent_flag": float(bool(_BAD_PERCENT.search(path + query))),
            "double_encoded_flag": float("%25" in (path + query).lower()),
            "null_byte_flag": float("\x00" in combined),
            "control_char_ratio": len(_CONTROL.findall(combined))
            / max(1, len(combined)),
            "path_entropy": entropy(
                normalized_target.split("?", 1)[0].encode("utf-8", "replace")
            ),
            "query_entropy": entropy(
                unquote_plus(query).encode("utf-8", "replace")
            ),
            "body_entropy": entropy(body),
            "target_entropy": entropy(combined.encode("utf-8", "replace")),
            "special_char_ratio": special_ratio(combined),
            "digit_ratio": char_ratio(combined, str.isdigit),
            "alpha_ratio": char_ratio(combined, str.isalpha),
            "has_sql_keyword": float(bool(_SQL.search(combined))),
            "has_xss_token": float(bool(_XSS.search(combined))),
            "has_traversal": float(bool(_TRAVERSAL.search(combined))),
            "has_command_token": float(bool(_COMMAND.search(combined))),
            "query_key_entropy": entropy(
                "\x00".join(query_keys).encode("utf-8", "replace")
            ),
            "body_utf8_replacement_ratio": body_text.count("\ufffd")
            / max(1, len(body_text)),
        }

        return FeatureVector(
            self.schema_version,
            {name: clamp01(value) for name, value in values.items()},
        )

    @staticmethod
    def _bounded_text(value: str, limit: int) -> str:
        return unicodedata.normalize("NFKC", value or "")[:limit]

    @staticmethod
    def _decode_body(body: bytes) -> str:
        return unicodedata.normalize(
            "NFKC", body.decode("utf-8", errors="replace")
        )

    @classmethod
    def _normalized_headers(cls, headers) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in headers.items():
            if len(normalized) >= cls.max_headers:
                break
            name = unicodedata.normalize("NFKC", str(key)).strip().lower()
            if name and name not in normalized:
                normalized[name] = unicodedata.normalize(
                    "NFKC", str(value)
                )[: cls.max_header_value_chars]
        return normalized


def _multi_decode(value: str, decoder, rounds: int = 3) -> str:
    previous = unicodedata.normalize("NFKC", value or "")
    for _ in range(rounds):
        decoded = unicodedata.normalize("NFKC", decoder(previous))
        if decoded == previous:
            break
        previous = decoded
    return previous


def normalize_target(path: str, query: str) -> str:
    """Decode path/query without changing path '+' into a space."""
    normalized_path = _multi_decode(path, unquote)
    normalized_query = _multi_decode(query, unquote_plus)
    return f"{normalized_path}?{normalized_query}" if query else normalized_path


def parse_query_safely(query: str):
    try:
        return (
            parse_qsl(
                query,
                keep_blank_values=True,
                strict_parsing=False,
                max_num_fields=ProductionHTTPFeatureExtractor.max_query_fields,
            ),
            False,
        )
    except ValueError:
        return [], True


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    size = len(data)
    value = -sum(
        (count / size) * math.log2(count / size)
        for count in counts.values()
    )
    return clamp01(value / 8.0)


def special_ratio(text: str) -> float:
    return sum(
        not char.isalnum() and not char.isspace() for char in text
    ) / max(1, len(text))


def char_ratio(text: str, predicate) -> float:
    return sum(predicate(char) for char in text) / max(1, len(text))


def cookie_count(cookie: str) -> int:
    return 0 if not cookie else len(
        [part for part in cookie.split(";") if "=" in part]
    )


def content_length_mismatch(headers: dict[str, str], actual_len: int) -> float:
    raw = headers.get("content-length")
    if raw is None:
        return 0.0
    try:
        declared = int(raw)
    except (TypeError, ValueError):
        return 1.0
    return float(declared != actual_len)


def clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
