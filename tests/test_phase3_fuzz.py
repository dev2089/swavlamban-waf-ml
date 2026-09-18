from __future__ import annotations

import random
import string

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


def test_randomized_http_inputs_are_nonfatal() -> None:
    rng = random.Random(20260918)
    extractor = ProductionHTTPFeatureExtractor()
    alphabet = string.ascii_letters + string.digits + "%?&=/+;:._-"
    for i in range(20_000):
        path = "/" + "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 512)))
        query = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 1024)))
        body = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 4096)))
        headers = {
            f"X-Test-{j}": "".join(rng.choice(string.printable) for _ in range(rng.randrange(0, 64)))
            for j in range(rng.randrange(0, 20))
        }
        request = RequestEnvelope(
            request_id=str(i),
            method=rng.choice(("GET", "POST", "PUT", "PATCH", "DELETE", "UNKNOWN")),
            scheme=rng.choice(("http", "https")),
            host="example.test",
            path=path,
            query=query,
            headers=headers,
            body=body,
        )
        vector = extractor.extract(request)
        assert vector.schema_version == "http-v2"
        assert all(0.0 <= value <= 1.0 for value in vector.values.values())
