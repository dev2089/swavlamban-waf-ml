from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def feature_stress() -> None:
    from waf.core.models import RequestEnvelope
    from waf.features.http_v2 import ProductionHTTPFeatureExtractor

    extractor = ProductionHTTPFeatureExtractor()
    started = time.perf_counter()
    for i in range(20_000):
        request = RequestEnvelope(
            request_id=str(i),
            method=("GET", "POST", "PUT", "PATCH")[i % 4],
            scheme=("http", "https")[i % 2],
            host="example.test",
            path=f"/api/resource/{i % 97}",
            query=f"q={i}&dup={i % 11}&dup={i % 7}",
            headers={
                "content-type": "application/json",
                "cookie": "a=1; b=2",
                "x-test": str(i),
            },
            body=b'{"value":"stable"}',
        )
        vector = extractor.extract(request)
        if vector.schema_version != "http-v2":
            raise AssertionError("wrong feature schema")
        if any(not 0.0 <= value <= 1.0 for value in vector.values.values()):
            raise AssertionError("feature value outside [0,1]")
    elapsed = time.perf_counter() - started
    print(
        f"feature_stress=20000 elapsed_s={elapsed:.6f} "
        f"req_per_sec={20000/elapsed:.2f}"
    )


def main() -> None:
    run(
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "waf",
        "tests",
        "run_proxy.py",
        "phase2_benchmark.py",
        "phase2_demo.py",
        "phase3_benchmark.py",
    )
    run(sys.executable, "-m", "pytest", "-q", "tests")
    feature_stress()
    run(sys.executable, "phase3_benchmark.py")
    run(sys.executable, "phase2_self_test.py")
    print("PHASE3_SELF_TEST=PASS")


if __name__ == "__main__":
    main()
