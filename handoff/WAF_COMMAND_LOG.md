# Phase 1 Command/Evidence Log

- Terminal clone attempt failed because outbound DNS/network access to GitHub was unavailable. GitHub connector was therefore used for repository control while the terminal remained the local lab.
- `python -m compileall -q waf tests` -> PASS.
- `python -m unittest discover -s tests -v` -> 13/13 PASS.
- `scripts/phase1_self_test.py` -> PASS.
- 100,000 mixed randomized HTTP-like requests -> ~23,882 decisions/sec.

All measurements apply only to the new Phase 1 core and are not end-to-end WAF claims.
