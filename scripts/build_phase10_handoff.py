"""Create a self-contained, audit-oriented Phase 10 handoff archive."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "WAF_PHASE10_FINAL_HANDOFF.zip"
SOURCE = ROOT / "WAF_PHASE10_FINAL_REPOSITORY.zip"
STAGE = ROOT / ".phase10_handoff_stage"
BASELINE = "b94e6aada97c009861e42f1fd6cd705f232a65bb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_source_archive() -> None:
    if SOURCE.exists():
        SOURCE.unlink()
    subprocess.run(["git", "archive", "--format=zip", "--output", str(SOURCE), "HEAD"], cwd=ROOT, check=True)


def stage_copy() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    # Complete working tree snapshot, excluding VCS internals and the temporary stage itself.
    for source in ROOT.iterdir():
        if source.name in {".git", ".phase10_handoff_stage", OUT.name}:
            continue
        target = STAGE / source.name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    # Add exact source archive explicitly.
    shutil.copy2(SOURCE, STAGE / SOURCE.name)


def main() -> int:
    final_sha = git("rev-parse", "HEAD")
    build_source_archive()
    stage_copy()
    files = []
    for path in sorted(STAGE.rglob("*")):
        if path.is_file():
            rel = path.relative_to(STAGE).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})

    diff = subprocess.check_output(["git", "diff", f"{BASELINE}...{final_sha}"], cwd=ROOT, text=True)
    (STAGE / "handoff" / "PHASE10_COMPLETE_GIT_DIFF.patch").write_text(diff, encoding="utf-8")
    files.append({"path": "handoff/PHASE10_COMPLETE_GIT_DIFF.patch", "bytes": (STAGE / "handoff/PHASE10_COMPLETE_GIT_DIFF.patch").stat().st_size, "sha256": sha256(STAGE / "handoff/PHASE10_COMPLETE_GIT_DIFF.patch")})

    manifest = {
        "format": "swavlamban-waf-phase10-auditor-handoff-v1",
        "project": "dev2089/swavlamban-waf-ml",
        "challenge": "Challenge 3 - ML-integrated open-source WAF",
        "branch": git("branch", "--show-current"),
        "final_commit_sha": final_sha,
        "baseline_commit_sha": BASELINE,
        "repository_url": "https://github.com/dev2089/swavlamban-waf-ml",
        "exact_reproduction": [
            "python -m pytest -q",
            "python -m compileall -q waf tests scripts",
            "python scripts/phase10_master_exam.py",
            "python scripts/build_phase10_submission.py",
            "python scripts/build_phase10_handoff.py",
        ],
        "artifact_categories": {
            "measured": ["phase10_demo_evidence.json", "phase10_load_evidence.json", "phase10_tls_evidence.json", "phase10_waf_enforcement_evidence.json", "phase10_rule_replay_evidence.json"],
            "simulated": ["versioned synthetic ML training/evaluation dataset", "synthetic positive/negative rule replay corpus"],
            "design_projection": ["horizontal scaling beyond local CI runner", "million-request capacity"],
            "externally_constrained": ["public certificate issuance/rotation", "Internet-scale distributed traffic"],
        },
        "files": files,
        "source_archive": {"path": SOURCE.name, "sha256": sha256(SOURCE), "bytes": SOURCE.stat().st_size},
        "note": "This manifest is about the final source/evidence snapshot, not an endorsement of claims. The independent auditor should rerun the listed commands and inspect the implementation.",
    }
    (STAGE / "handoff" / "PHASE10_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (STAGE / "handoff" / "PHASE10_AUDIT_MANIFEST.md").write_text(
        "# Swavlamban WAF Phase 10 Auditor Manifest\n\n"
        f"Final commit: `{final_sha}`\n\n"
        "This handoff is self-contained. Start with `WAF_PROJECT_STATE.json`, then `handoff/START_HERE.md`, `handoff/PHASE10_FINAL_STATUS.md`, the master exam, the requirement traceability and the negative-evidence register.\n\n"
        "The independent auditor must execute the reproduction commands and treat all PASS labels as untrusted until independently reproduced.\n",
        encoding="utf-8",
    )

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE).as_posix())
    print(json.dumps({"archive": OUT.name, "bytes": OUT.stat().st_size, "sha256": sha256(OUT), "final_commit_sha": final_sha, "file_count": len(files) + 2}, indent=2, sort_keys=True))
    shutil.rmtree(STAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
