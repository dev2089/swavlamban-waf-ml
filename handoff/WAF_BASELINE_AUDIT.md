# Baseline Audit Snapshot

Source: `dev2089/swavlamban-waf-ml`, main commit `1cc4f91dd6828039f834ae4dc2b466191d04f229`.

## Conclusion
The repository was a prototype/demo foundation, not production-ready. Documentation over-claimed capabilities compared with implementation.

## Key findings

- `backend/server.py`: request simulation via `/api/analyze`, not real traffic interception or enforcement.
- `ml_model.py`: Isolation Forest + autoencoder on synthetic numerical data; not credible evidence of the documented ML system.
- `rule_generator.py`: static keyword-based rule generation, not a complete ML feedback/recommendation lifecycle.
- Supabase migration: broad public/anon policies despite RLS; poor production security boundary.
- Raw request logging risks storing sensitive request data.
- Dashboard analytics performed broad client-side queries; some dashboard changes were hardcoded.
- Rules UI and storage policies did not fully match advertised functionality.
- `app.py`: separate demo Flask path with hardcoded metrics/placeholder analysis.
- `main.py`: orchestration stubs/TODOs; not the active runtime path.
- `start.sh`: development startup helper, not production supervision.
- Requirements/docs described infrastructure and capabilities absent from the observed tree.

## Evidence limitation
The large `package-lock.json` was not available as one complete tool response during the audit. The audit does not claim a full line-by-line package-lock review.
