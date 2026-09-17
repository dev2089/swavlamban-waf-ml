from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    py_files = list((ROOT / "waf").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    for path in py_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = re.compile(r"(?:sk-[A-Za-z0-9]|api[_-]?key\s*=|password\s*=|secret[_-]?key\s*=)", re.I)
    network_words = re.compile(r"(?:requests\.|httpx|urllib\.request|socket\.socket|subprocess\.(run|Popen|call))")
    for path in (ROOT / "waf").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), f"secret-like content in {path}"
        assert not network_words.search(text), f"external I/O found in {path}"
    env = dict(PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, env=env, text=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
