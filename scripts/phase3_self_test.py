from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args))
    result = subprocess.run(args, cwd=ROOT, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def benchmark() -> None:
    from waf.core.models import RequestEnvelope
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor

    extractor = ProductionHTTPFeatureExtractor()
    requests = [
        RequestEnvelope(str(i), "GET" if i % 3 else "POST", "https", "example.test", f"/api/item/{i % 31}", f"q={i}")
        for i in range(20_000)
    ]
    started = time.perf_counter()
    for request in requests:
        extractor.extract(request)
    elapsed = time.perf_counter() - started
    print(f"benchmark: {len(requests)} extractions in {elapsed:.4f}s ({len(requests)/elapsed:.1f} req/s), features={len(extractor.extract(requests[0]).values)}")


if __name__ == "__main__":
    run(sys.executable, "-m", "compileall", "-q", "waf", "tests")
    run(sys.executable, "-m", "pytest", "-q", "tests/test_features.py", "tests/test_pipeline.py", "tests/test_phase2.py", "tests/test_phase3_features.py", "tests/test_phase3_edge.py")
    benchmark()
    print("PHASE3 SELF-TEST: PASS")
