"""Fail the release when legacy prototype code/claims leak into current surfaces."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

CURRENT_SURFACES = [
    ROOT / "README.md",
    ROOT / "START_HERE.md",
    ROOT / "FEATURES.md",
    ROOT / "QUICKSTART.md",
    ROOT / "QUICKSTART_NEW.md",
    ROOT / "TECHNICAL_DOCUMENTATION.md",
    ROOT / "WAF_ML_SETUP.md",
    ROOT / "app.py",
    ROOT / "main.py",
    ROOT / "setup.sh",
    ROOT / "requirements.txt",
    ROOT / "backend" / "requirements.txt",
]


def main() -> int:
    failures: list[str] = []

    app = (ROOT / "app.py").read_text(encoding="utf-8")
    if "from flask import" in app or "dashboard_data" in app or "Potential Attack Vector" in app:
        failures.append("app.py still contains the legacy Flask/demo application")
    if "from waf.api.production_api import app" not in app:
        failures.append("app.py does not delegate to the canonical FastAPI application")

    main_py = (ROOT / "main.py").read_text(encoding="utf-8")
    if "WAFMLOrchestrator" in main_py or "TODO:" in main_py or '"predictions": None' in main_py:
        failures.append("main.py still contains obsolete orchestration stubs")
    if "waf.gateway.proxy" not in main_py:
        failures.append("main.py does not delegate to the canonical gateway")

    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("flask", "tensorflow", "seaborn", "pandas==1.5"):
        if re.search(rf"^\s*{re.escape(package)}", req, flags=re.MULTILINE):
            failures.append(f"requirements.txt contains obsolete dependency: {package}")

    backend_req = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("flask", "tensorflow", "pandas==1.5"):
        if re.search(rf"^\s*{re.escape(package)}", backend_req, flags=re.MULTILINE):
            failures.append(f"backend/requirements.txt contains obsolete dependency: {package}")

    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    for fragment in ("notebooks", 'directories=("data"', "requirements-dev.txt"):
        if fragment in setup:
            failures.append(f"setup.sh contains obsolete setup behavior: {fragment}")
    if "pip install -r requirements.txt" not in setup:
        failures.append("setup.sh does not install the canonical requirements.txt")
    if "python -m pytest -q" not in setup:
        failures.append("setup.sh does not execute the regression suite")

    forbidden_claims = (
        "96.5%",
        "95.2%",
        "97.8%",
        "1247",
        "production-ready security platform",
        "production-ready scalable",
        "CNN_MODEL_VERSION=v2.3.1",
        "requests_processed\": 1500000",
    )
    for path in CURRENT_SURFACES:
        text = path.read_text(encoding="utf-8", errors="replace")
        for claim in forbidden_claims:
            if claim in text:
                failures.append(f"legacy/unverified claim '{claim}' remains in {path.relative_to(ROOT)}")

    runtime_files = list((ROOT / "waf").rglob("*.py")) + [ROOT / "app.py", ROOT / "main.py", ROOT / "backend" / "server.py", ROOT / "run_proxy.py"]
    for path in runtime_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "TODO:" in text or "FIXME:" in text:
            failures.append(f"TODO/FIXME remains in runtime source: {path.relative_to(ROOT)}")

    result = "PASS" if not failures else "FAIL"
    print(f"phase10 repository hygiene: {result}")
    if failures:
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
