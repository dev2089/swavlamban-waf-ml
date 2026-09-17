#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v python3 >/dev/null || { echo "Python 3.11+ is required." >&2; exit 1; }
command -v git >/dev/null || { echo "Git is required." >&2; exit 1; }

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
PY

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ -f .env.example && ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Review secrets/configuration before production use."
fi

python -m compileall -q waf tests scripts
python -m pytest -q

echo
printf '%s\n' "Setup and regression tests completed successfully."
printf '%s\n' "Canonical runtime: python -m waf.gateway.proxy"
printf '%s\n' "API/dashboard runtime: python backend/server.py"
printf '%s\n' "Full release gate: python scripts/phase10_master_exam.py"
