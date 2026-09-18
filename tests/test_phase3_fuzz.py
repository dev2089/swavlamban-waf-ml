from __future__ import annotations

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


def test_randomized_http_inputs_are_nonfatal() -> None:
    # The complete 20,000-input fuzz campaign is run by phase3_fuzz.py.
    # Keep the standard pytest regression bounded for fast iteration.
    total = 2_000
    extractor = ProductionHTTPFeatureExtractor()
    for i in range(total):
        request = RequestEnvelope(
            request_id=str(i),
            method=("GET", "POST", "PUT", "PATCH", "DELETE", "UNKNOWN")[i % 6],
            scheme=("http", "https")[i % 2],
            host="example.test",
            path=f"/fuzz/{i % 97}/%25{i % 13}",
            query=f"q={i}&a={i % 7}&a={i % 11}",
            headers={"x-fuzz": str(i), "content-type": "application/json"},
            body=(b"abc%ff" + bytes([i % 256])) * (i % 16),
        )
        vector = extractor.extract(request)
        assert vector.schema_version == "http-v2"
        assert all(0.0 <= value <= 1.0 for value in vector.values.values())
