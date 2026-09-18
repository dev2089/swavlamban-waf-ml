# Durable Project Memory

This directory is the compact, machine-readable and human-readable memory for future ChatGPT conversations.

Always read the current `PHASE_*_STATE.json`, `PHASE_*_INDEPENDENT_AUDIT.md`, `PROJECT_JOURNAL.jsonl`, and `FUTURE_CHAT_START_HERE.md` before modifying the project.

Rules:
1. Source code and executable evidence are authoritative.
2. Status text is not evidence.
3. Every phase records completed work, tests, failures, fixes and remaining work.
4. Final project release requires 100% gate and no critical defect.
5. Never convert unavailable or unexecuted tests into PASS.