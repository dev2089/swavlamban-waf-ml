"""Compatibility entry point for the comprehensive Phase 10 audit generator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import phase10_audit_v2 as _audit

_original_tracked_files = _audit.tracked_files


def _tracked_without_audit_sources() -> list[str]:
    excluded = {"scripts/phase10_audit.py", "scripts/phase10_audit_v2.py"}
    return [path for path in _original_tracked_files() if path not in excluded]


_audit.tracked_files = _tracked_without_audit_sources


if __name__ == "__main__":
    raise SystemExit(_audit.main())
