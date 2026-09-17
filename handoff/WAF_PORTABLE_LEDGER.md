# Portable Project Ledger

`state/project_ledger.db` is the project's execution/state database. It is deliberately separate from the future WAF runtime traffic database.

## What is recorded
- project metadata and current phase;
- phase status and scores;
- phase tasks with evidence references;
- every recorded test cycle, including failures and remediation;
- change history;
- artifact paths, versions, SHA-256 and sizes;
- environment snapshots used for reproducibility.

## New-chat workflow
1. Read `handoff/START_HERE.md`.
2. Read `WAF_PROJECT_STATE.json`.
3. Read `handoff/WAF_MASTER_PLAN.md` and the latest phase test/exam documents.
4. Read the phase execution log, changelog and command log.
5. Open `state/project_ledger.db` for the chronological evidence ledger.
6. Inspect the executable code before changing state.

The ledger is evidence/state, not a substitute for rerunning tests.
