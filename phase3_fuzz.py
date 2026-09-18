from __future__ import annotations

import random
import string
import time

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


def main() -> None:
    rng = random.Random(20260918)
    extractor = ProductionHTTPFeatureExtractor()
    alphabet = string.ascii_letters + string.digits + "%?&=/+;:._-"
    total = 20_000
    started = time.perf_counter()

    for i in range(total):
        path = "/" + "".join(rng.choice(alphabet) for _ in range(rng.randrange(128)))
        query = "".join(rng.choice(alphabet) for _ in range(rng.randrange(256)))
        body = bytes(rng.randrange(256) for _ in range(rng.randrange(512)))
        headers = {
            f"X-Test-{j}": "".join(
                rng.choice(string.printable) for _ in range(rng.randrange(24))
            )
            for j in range(rng.randrange(8))
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
        if vector.schema_version != "http-v2":
            raise AssertionError("wrong schema")
        if any(not 0.0 <= value <= 1.0 for value in vector.values.values()):
            raise AssertionError("feature outside [0,1]")

    elapsed = time.perf_counter() - started
    print(
        f"fuzz_inputs={total} exceptions=0 elapsed_s={elapsed:.3f} "
        f"inputs_per_sec={total/elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
