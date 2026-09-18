from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


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
        "phase3_fuzz.py",
    )
    run(sys.executable, "-m", "pytest", "-q", "tests")
    run(sys.executable, "phase3_fuzz.py")
    run(sys.executable, "phase3_benchmark.py")
    run("bash", "tests/test_nginx_integration.sh")
    print("PHASE3_SELF_TEST=PASS")


if __name__ == "__main__":
    main()
